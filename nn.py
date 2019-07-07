import spacy,csv, nltk, re,sys, pickle
from spacymoji import Emoji
from nltk.util import trigrams, bigrams
from nltk.tokenize import RegexpTokenizer
from numpy import array
from keras.preprocessing.text import one_hot
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Bidirectional, Input, Conv1D, Flatten,\
    MaxPooling1D, GRU, SpatialDropout1D
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import html
from sklearn.utils.class_weight import compute_class_weight
#plt.style.use('ggplot')


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 794#1913#423#743
EPOCHS = 100


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average attention mechanism from:
        Zhou, Peng, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao and Bo Xu.
        “Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.”
        ACL (2016). http://www.aclweb.org/anthology/P16-2034
    How to use:
    see: [BLOGPOST]
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]

        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True)  # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_val, y_pred):
    #y_true, y_pred = check_units(y_true, y_pred)
    #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #precision = true_positives / (predicted_positives + K.epsilon())
    tp = 0
    fp = 0
    fn = 0



    for i in range(0,len(y_val)):


        predicted_label = y_pred[i]

        real_label = y_val[i]

        if real_label == predicted_label:
            tp+=1
        else:
            if real_label:
                fn+=1
            else:
                fp+=1
        #print('Actual label:' + str(real_label))
        #print("Predicted label: " + str(predicted_label))


    precision = tp/(tp+fp)

    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def clean_tweet(tweet_text):
	tweet_text = tweet_text.replace('—', " ")#.replace("'", "’")
	tweet_text = ' '.join(tweet_text.split())
	return tweet_text.strip()



def text_processing(nlp, text):
    tweet_text = clean_tweet(text)

    tweet_text = tweet_text.replace("'", "&&&").replace("’", "&&&").replace("-", "&#&")

    doc = nlp(tweet_text)
    print(text)

    for token in doc:
        print(str(token).replace("&&&", "'").replace("&#&","-"))


def get_word_index_from_file(path):
    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")
    word_index = {}
    for row in read:
        word_index[row["word"]] = int(row["index"])
    fd.close()
    return word_index


def replace_tweet(tweet_text):

    tweet_text = html.unescape(tweet_text)

    return tweet_text.replace("'", "QUOTE_SYMBOL").replace("‘", "QUOTE_SYMBOL").replace("’", "QUOTE_SYMBOL").replace("-", "HYPH_SYMBOL").replace(";", " ").replace("#", "HASHTAG_SYMBOL")

def unreplace_tweet(tweet_text):

    return tweet_text.replace("QUOTE_SYMBOL", "'").replace("HYPH_SYMBOL", "-").replace("HASHTAG_SYMBOL", "#").replace("EMOJI_SYMBOL","#&").lower()


def get_preprocessed_tweets(path, nlp, word_index, label_type="subtask_a", label_dict= {"OFF": 1}, max_instances=0):

    print(path)

    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")

    texts = []

    labels = []

    i=0

    for row in read:

        sent = []
        i += 1
        tweet_text = clean_tweet(row["tweet"])
        if "test" in label_type:
            label=row["id"]
        else:
            label = label_dict.get(row[label_type], 0)
        labels.append(label)
        tweet_text =replace_tweet(tweet_text)

        doc = nlp(tweet_text)

        for token in doc:
            word = unreplace_tweet(str(token))
            sent.append(word)

        texts.append(sent)
        if i == max_instances:
            break

    sequences = []
    for sent in texts:
        seq = []
        for word in sent:
            i = word_index.get(word, 0)
            if i:
                seq.append(i)
        sequences.append(seq)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    if not "test" in label_type:
        labels = to_categorical(np.asarray(labels))
    fd.close()
    return data, labels


def get_preprocessed_labels(path, label_type="subtask_a", label_dict= {"OFF": 1}):
    fd = open(path, "r")

    read = csv.DictReader(fd, dialect="excel-tab")

    labels = []

    for row in read:

        label = label_dict.get(row[label_type], 0)
        labels.append(label)

    return labels




def get_word_index(path, nlp):

    fd = open(path, "r")
    read = csv.DictReader(fd, dialect="excel-tab")
    word_set = set()
    i=0
    for row in read:
        i += 1
        tweet_text = clean_tweet(row["tweet"])
        tweet_text = replace_tweet(tweet_text)
        doc = nlp(tweet_text)
        for token in doc:
            word = unreplace_tweet(str(token))
            word_set.add(word)

    word_index = {}
    i = 0
    for tok in word_set:
        i += 1
        word_index[tok] = i

    fd.close()
    return word_index

def get_pretrained_embedding(emb_path, word_index):
    embeddings_index = {}
    f = open(emb_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    #embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    embedding_matrix = np.random.uniform(-0.8, 0.8, (len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def lstm_simple(embedding_matrix):
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    opt = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(2, activation='sigmoid'))
    print(model.summary())
    # try using different optimizers and different optimizer configs
    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lstm_simple_binary(embedding_matrix):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    #model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # try using different optimizers and different optimizer configs
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])


    return model

def cnn(embedding_matrix):
    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    return model

def lstm_simple_binary_attent(embedding_matrix):
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


    #Bidirectional(LSTM(units=rnn_units, return_sequences=True, dropout=d,
    #                   recurrent_dropout=rd)
    d = 0.5
    rd = 0.5
    rnn_units = 100
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(d))
    model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True)))#, dropout=d, recurrent_dropout=rd)))
    model.add(AttentionWeightedAverage())
    model.add(Dropout(d))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def RNN2(embedding_matrix):
    opt = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_embedding_matrix(emb_path, model):

    word_index = {}
    for word in model.vocab:
        word_index[word] = model.vocab[word].index
        print(word)
        print(model.vocab[word].index)

    embeddings_index = {}
    f = open(input_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 300

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


def plot_history(history):
    history_dict = history.history
    print(history_dict.keys())

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def save_obj(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()

def load_obj(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj



def prepocess_organizers_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    label_tranf = {"CAG": "OFF",
                   "OAG": "OFF",
                   "NAG": "NOT"}
    out = open(add_proc_path, "w")
    # out.write("id\ttweet")
    with open(add_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweet = row['tweet'].replace("\n", " ")

            total += 1
            if len(tweet.split(" ")) > 55:
                continue
            if tweet.startswith("RT"):
                continue
            if " RT " in tweet:
                continue
            if not len(tweet):
                continue
            count += 1
            #print(row['id'], row['class'], tweet)

            tweet = re.sub(r"http\S+", "URL", tweet)
            tweet = re.sub('@[^\s]+', '@USER', tweet)

            print(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']])

            out.write(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']] + "\n")

    print(total)
    print(count)
    out.close()

def prepocess_additional_dataset(add_path, add_proc_path):
    count = 0
    total = 0

    label_tranf = {"0": "OFF",
                   "1": "OFF",
                   "2": "NOT"}
    out = open(add_proc_path, "w")
    # out.write("id\ttweet")
    with open(add_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tweet = row['tweet'].replace("\n", " ")
            total += 1
            if tweet.startswith("RT"):
                continue
            if " RT " in tweet:
                continue
            if not len(tweet):
                continue
            count += 1
            #print(row['id'], row['class'], tweet)

            tweet = re.sub(r"http\S+", "URL", tweet)
            tweet = re.sub('@[^\s]+', '@USER', tweet)

            print(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']])

            out.write(row['id'] + "\t" + tweet + "\t" + label_tranf[row['class']] + "\n")

    print(total)
    print(count)
    out.close()

if __name__ == '__main__':

    print("jje")

    #test = "I swear nobody actually cares about Valentine's day unless they are single &amp; jaded and want to bitch. &#10060;&#11093;&#65039;&#10060;&#11093;&#65039;&#127801; hahaha."
    #print(html.unescape(test))
    #re.finditer('&#([0-9]+);')




    add_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/organizers_data.csv"
    add_proc_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/organizers_data_processed.tsv"
    #prepocess_organizers_dataset(add_path, add_proc_path)



    add_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/labeled_data.csv"
    add_proc_path = "/home/upf/corpora/SEMEVAL19_Task6/additional/labeled_data_processed.tsv"
    #prepocess_additional_dataset(add_path, add_proc_path)
    #sys.exit()

    max_instances = 0



    categorical=True

    res_name = "_org_rmsprop_001"

    SAVE_MODE = False

    version = ""

    if SAVE_MODE:

        nlp = spacy.load('en')
        emoji = Emoji(nlp)
        nlp.add_pipe(emoji, first=True)


    print("NLP Loaded")
    #"""


    vocab_path = "/home/upf/corpora/SEMEVAL19_Task6/train_and_test"+version+".tsv"
    word_index_pkl_path = "/home/upf/corpora/SEMEVAL19_Task6/word_index_train_test" + version + ".pkl"

    if SAVE_MODE:
        word_index = get_word_index(vocab_path, nlp)

        word_index_path = "/home/upf/corpora/SEMEVAL19_Task6/word_index_train_test"+version+".tsv"

        out = open(word_index_path, "w")
        out.write("word" + "\t" + "index" + "\n")
        for k in sorted(word_index):
            out.write(k + "\t" + str(word_index[k]) + "\n")
        out.close()


        save_obj(word_index, word_index_pkl_path)

    word_index = load_obj(word_index_pkl_path)


    print(len(word_index))
    print("Word Index Loaded")

    embedding_matrix_path = "/home/upf/corpora/SEMEVAL19_Task6/embedding_matrix" + version + ".pkl"
    if SAVE_MODE:

        emb_path = "/home/upf/Downloads/model_swm_300-6-10-low.w2v"
        embedding_matrix = get_pretrained_embedding(emb_path, word_index)
        save_obj(embedding_matrix, embedding_matrix_path)

    embedding_matrix = load_obj(embedding_matrix_path)

    print(embedding_matrix.shape)
    print("Embbeding Matrix Created")

    data_path = "/home/upf/corpora/SEMEVAL19_Task6/data" + version + ".pkl"
    labels_path = "/home/upf/corpora/SEMEVAL19_Task6/labels" + version + ".pkl"
    if SAVE_MODE:
        train_file = "/home/upf/corpora/SEMEVAL19_Task6/offenseval-training-v1"+version+".tsv"
        data, labels = get_preprocessed_tweets(train_file, nlp, word_index, "subtask_a", {"OFF":1})
        save_obj(data, data_path)
        save_obj(labels, labels_path)

    data = load_obj(data_path)
    print(data.shape)



    if categorical:
        labels = load_obj(labels_path)
    else:
        labels = get_preprocessed_labels(train_file, label_type="subtask_a", label_dict={"OFF": 1})



    total_lines = data.shape[0]
    test_count = int(total_lines * 0.15)

    x_train = data[test_count:]
    y_train = labels[test_count:]
    x_val = data[:test_count]
    y_val = labels[:test_count]

    x_train=data
    y_train = labels
    """
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)
    """
    print("Training Set Processed")


    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=2, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    # train the model

    #model = lstm_simple(embedding_matrix)

    #model = RNN2(embedding_matrix)

    #model = lstm_simple_binary(embedding_matrix)

    model = lstm_simple_binary_attent(embedding_matrix)

    #model = cnn(embedding_matrix)


    history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.15,
              callbacks=callbacks_list,
              shuffle=True)

    #plot_model_history(history)

    #accr = model.evaluate(x_val, y_val)

    """
    tp=0
    fp=0
    fn=0
    for i in range(0,len(x_val)):
        prediction = model.predict(np.array([x_val[i]]))

        if categorical:

            predicted_label = np.argmax(prediction[0])

            real_label = np.argmax(y_val[i])
        else:
            predicted_label = 0
            if prediction[0] >.5:
                predicted_label = 1

            real_label = y_val[i]

        if real_label == predicted_label:
            tp+=1
        else:
            if real_label:
                fn+=1
            else:
                fp+=1
        #print('Actual label:' + str(real_label))
        #print("Predicted label: " + str(predicted_label))


    print ("TP\tFP\tFN\tPRECISION\tRECALL\tFMEASURE:")
    p = tp/(tp+fp)
    r=tp / (tp + fn)
    f = (2*p*r)/(p+r)
    print (str(tp) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(p) + "\t" + str(r) + "\t" + str(f))

    """
    #Testing

    print("Testing....")

    testing_path = "/home/upf/corpora/SEMEVAL19_Task6/testset-taska.tsv"
    #data_test, ids_test = get_preprocessed_tweets(testing_path, nlp, word_index, "test")

    data_path = "/home/upf/corpora/SEMEVAL19_Task6/data_test"+version+".pkl"
    #save_obj(data_test, data_path)

    ids_test_path = "/home/upf/corpora/SEMEVAL19_Task6/ids_test_a"+version+".pkl"
    #save_obj(ids_test, ids_test_path)

    data_test = load_obj(data_path)
    print(data_test.shape)
    ids_test = load_obj(ids_test_path)

    i = 0

    label_test_dict = {0:"NOT",
                       1:"OFF"}

    result_path = "/home/upf/corpora/SEMEVAL19_Task6/task_a_result"+res_name+".csv"
    out = open(result_path, "w")

    while i < len(ids_test):
        prediction = model.predict(np.array([data_test[i]]))

        if categorical:
            predicted_label = np.argmax(prediction[0])
        else:
            predicted_label = 0
            if prediction[0] >.5:
                predicted_label = 1
        #print(ids_test[i] + "\t" + str(label_test_dict[predicted_label]))
        out.write(ids_test[i] + "\t" + str(label_test_dict[predicted_label]) + "\n")
        i += 1
    out.close()
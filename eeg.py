import pandas as pd 
import json
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np

df = pd.read_csv("/Users/wnowak/Desktop/eeg_data.csv")

df['eeg_power'] = df.eeg_power.map(json.loads)


# drop unused features. just leave eeg_power and the label
df = df.drop('Unnamed: 0', 1)
df = df.drop('id', 1)
df = df.drop('indra_time', 1)
df = df.drop('browser_latency', 1)
df = df.drop('reading_time', 1)
df = df.drop('attention_esense', 1)
df = df.drop('meditation_esense', 1)
df = df.drop('raw_values', 1)
df = df.drop('signal_quality', 1)
df = df.drop('createdAt', 1)
df = df.drop('updatedAt', 1)


# separate eeg power to multiple columns
to_series = pd.Series(df['eeg_power']) # df to series
eeg_features=pd.DataFrame(to_series.tolist()) #series to list and then back to df

df = pd.concat([df,eeg_features], axis=1, join='outer') # concatenate the create columns
df = df.drop('eeg_power', 1) # drop comma separated cell


# prepare for training
label=df.pop("label") # pop off labels to new group
print df.shape
print(df.head())
# convert to np array. df has our featuers
df=df.values



# convert labels to onehots 
train_labels = pd.get_dummies(label)
print(train_labels)
# make np array
train_labels = train_labels.values
print(train_labels.shape)

x_train,x_test,y_train,y_test = train_test_split(df,train_labels,test_size=0.2)
# so now we have predictors and y values, separated into test and train

x_train,x_test,y_train,y_test = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'),np.array(y_train,dtype='float32'),np.array(y_test,dtype='float32')


# there are 8 features
# place holder for inputs. feed in later
x = tf.placeholder(tf.float32, [None, 8])
# # # take 20 features  to 10 nodes in hidden layer
w1 = tf.Variable(tf.random_normal([8, 1000],stddev=.5,name='w1'))
# # # add biases for each node
b1 = tf.Variable(tf.zeros([1000]))
# # calculate activations 
hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)
w2 = tf.Variable(tf.random_normal([1000, 69],stddev=.5,name='w2'))
b2 = tf.Variable(tf.zeros([69]))

# # placeholder for correct values 
y_ = tf.placeholder("float", [None,69])
# # #implement model. these are predicted ys
y = tf.nn.softmax(tf.matmul(hidden_output, w2) + b2)


loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, y_, name='xentropy')))
opt = tf.train.AdamOptimizer(learning_rate=.005)
train_step = opt.minimize(loss, var_list=[w1,b1,w2,b2])

def get_mini_batch(x,y):
	rows=np.random.choice(x.shape[0], 100)
	return x[rows], y[rows]

# start session
sess = tf.Session()
# init all vars
init = tf.initialize_all_variables()
sess.run(init)

ntrials = 1000
for i in range(ntrials):
    # get mini batch
    a,b=get_mini_batch(x_train,y_train)
    # run train step, feeding arrays of 100 rows each time
    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})
    if i%100 ==0:
    	print("epoch is {0} and cost is {1}".format(i,cost))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})))

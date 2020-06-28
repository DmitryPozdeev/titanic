import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_data = pd.read_csv('input_titanic/train.csv').fillna(0)
test_data = pd.read_csv('input_titanic/test.csv').fillna(0)

g = sns.catplot(x="Parch", y="Survived", data=train_data,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")

sns.countplot( x = 'Survived', data = train_data)

sns.countplot( x = 'Survived', hue = 'Sex', data = train_data)

sns.countplot( x = 'Survived', hue = 'Pclass', data = train_data)

train_data[['Sex']] = train_data[['Sex']].replace('male', 1).replace('female', 0)
train_data[['Pclass']] = train_data[['Pclass']]
train_data[['Age']] = train_data[['Age']] / 100
train_data[['Fare']] = train_data[['Fare']] / 100
train_data[['Parch']] = train_data[['Parch']] / 10
train = train_data[['Pclass', 'Sex', 'Parch', 'Age','Fare']]
test = train_data[['Survived']]

x_train = train.to_numpy()[:800]
x_test = test.to_numpy()[:800]

y_train = train.to_numpy()[800:]
y_test = test.to_numpy()[800:]

test_data[['Sex']] = test_data[['Sex']].replace('male', 1).replace('female', 0)
test_data[['Pclass']] = test_data[['Pclass']]
test_data[['Age']] = test_data[['Age']] / 100
test_data[['Fare']] = test_data[['Fare']] / 100
test_data[['Parch']] = test_data[['Parch']] / 10

export_test_data = test_data[['Pclass', 'Sex', 'Parch', 'Age','Fare']].to_numpy()
export_ids = test_data[['PassengerId']].to_numpy()

print(export_test_data[:10])

with np.printoptions(threshold=np.inf):
    print(x_train[:15])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(250, activation=tf.nn.relu),
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, x_test, epochs=150)

model.evaluate(y_train, y_test)

result = tf.argmax(model.predict(export_test_data),1).numpy()

result = np.array(result, dtype=np.int)
result = result.reshape(418, 1)
structuredArr = np.concatenate((export_ids, result), axis=1)

np.savetxt('submition.csv', structuredArr, delimiter=',', fmt=['%i' , '%i'], header='PassengerId,Survived', comments='')
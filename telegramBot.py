import requests
import tensorflow as tf

# token = ''
# ping_url = 'https://api.telegram.org/bot' + str(token) + '/getUpdates'
# response = requests.get(ping_url).json()
# print(response)

# chat_id = response['result'][0]['message']['chat']['id']

# ping_url = 'https://api.telegram.org/bot' + str(token) + '/sendMessage?' + \
#            'chat_id=' + str(chat_id) + \
#            '&parse_mode=Markdown' + \
#            '&text=' + 'first+message+from+bot'
# response = requests.get(ping_url)
# print(ping_url)

global total_epochs


class BotCallback(tf.keras.callbacks.Callback):
    def __init__(self, personal_token, test_ds):
        self.personal_token = personal_token
        self.ping_url = 'https://api.telegram.org/bot' + str(self.personal_token) + '/getUpdates'
        self.response = requests.get(self.ping_url).json()
        self.chat_id = self.response['result'][0]['message']['chat']['id']
        self.test_data = test_ds

    def send_message(self, message):
        self.ping_url = 'https://api.telegram.org/bot' + str(self.personal_token) + '/sendMessage?' + \
                        'chat_id=' + str(self.chat_id) + \
                        '&parse_mode=Markdown' + \
                        '&text=' + message
        self.response = requests.get(self.ping_url)

    def on_epoch_end(self, epoch, logs):
        message = "Epoch: {}\nTrain Loss: {:.4f} - Train Acc: {:.4f}\nValidation Loss: {:.4f} - Validation Acc: {:.4f}".format(
            epoch + 1,
            logs["loss"],
            logs["accuracy"],
            logs["val_loss"],
            logs["val_accuracy"])
        self.send_message(message)

    def on_train_end(self, logs=None):
        test_loss, test_acc = self.model.evaluate(self.test_data)
        message = f"Final epoch with test accuracy: {test_acc}"
        self.send_message(message)

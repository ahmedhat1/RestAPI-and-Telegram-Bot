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
    def __init__(self, personal_token):
        self.personal_token = personal_token
        self.ping_url = 'https://api.telegram.org/bot' + str(self.personal_token) + '/getUpdates'
        self.response = requests.get(self.ping_url).json()
        self.chat_id = self.response['result'][0]['message']['chat']['id']

    def send_message(self, message):
        self.ping_url = 'https://api.telegram.org/bot' + str(self.personal_token) + '/sendMessage?' + \
                        'chat_id=' + str(self.chat_id) + \
                        '&parse_mode=Markdown' + \
                        '&text=' + message
        self.response = requests.get(self.ping_url)

    def on_epoch_end(self, epoch, logs):
        if epoch + 1 == total_epochs:
            message = "final epoch with Validation Acc:{:.4f}".format(
                logs["val_accuracy"]
            )
        else:
            message = "Epoch: {}\nTrain Loss: {:.4f} - Train Acc: {:.4f}\nValidation Loss: {:.4f} - Validation Acc: {:.4f}".format(
                epoch + 1,
                logs["loss"],
                logs["accuracy"],
                logs["val_loss"],
                logs["val_accuracy"]
            )
        self.send_message(message)

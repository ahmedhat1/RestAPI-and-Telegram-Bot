import requests
import tensorflow as tf

personal_token = ''
ping_url = 'https://api.telegram.org/bot'+str(personal_token)+'/getUpdates'
response = requests.get(ping_url).json()
print(response)

chat_id = response['result'][0]['message']['chat']['id']

ping_url = 'https://api.telegram.org/bot'+str(personal_token)+'/sendMessage?'+\
                    'chat_id='+str(chat_id)+\
                    '&parse_mode=Markdown'+\
                    '&text='+ 'first+message+from+bot'
response = requests.get(ping_url)
print(ping_url)

# class BotCallback(tf.keras.callbacks.Callback):
#     def __init__(self,personal_token):
#         self.personal_token = personal_token
#         self.ping_url = 'https://api.telegram.org/bot'+str(self.personal_token)+'/getUpdates'
#         self.response = requests.get(self.ping_url).json()
#         self.chat_id = self.response['result'][0]['message']['chat']['id']
#
#     def send_message(self,message):
#         self.ping_url = 'https://api.telegram.org/bot'+str(self.personal_token)+'/sendMessage?'+\
#                         'chat_id='+str(self.chat_id)+\
#                         '&parse_mode=Markdown'+\
#                         '&text='+message
#         self.response = requests.get(self.ping_url)
#
#     def on_epoch_end(self, epoch, epoch_logs):
#         # TODO
#         pass
import json
import requests

HOST = '123.125.8.44'
PORT = 5014


class RASAClient:
    def __init__(self, host=HOST, port=PORT, username='me', password='rasarasa'):
        self.host = host
        self.port = port
        self.token = self.get_token(username, password)

    def get_token(self, username, password):
        token = json.loads(
            requests.post(f'http://{self.host}:{self.port}/api/auth',
                          json={
                              "username": username,
                              "password": password
                          }).text)['access_token']
        token = f'Bearer {token}'
        return token

    def send_message(self, sender_id, message):
        response = requests.post(f'http://{self.host}:{self.port}/api/conversations/{sender_id}/messages',
                                 json={"message": message},
                                 headers={"Authorization": self.token})
        return response.text

    def send_nlu_data(self, text, intent):
        response = requests.post(f'http://{self.host}:{self.port}/api/projects/default/training_examples',
                                 json={
                                     "examples": [{
                                         "team": "default",
                                         "project_id": "default",
                                         "text": text,
                                         "intent": intent,
                                         "entities": []
                                     }]
                                 },
                                 headers={"Authorization": self.token})
        return response.text

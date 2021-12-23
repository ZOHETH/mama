import json
import requests
import logging

import sqlite3

HOST = '123.125.8.44'
PORT = 5014


class RASAClient:
    def __init__(self, host=HOST, port=PORT, username='me', password='rasarasa', db_path=None):
        self.url = f'http://{host}:{port}'
        self.token = self.get_token(username, password)
        self.db_path = db_path

    def get_token(self, username, password):
        token = json.loads(
            requests.post(f'{self.url}/api/auth',
                          json={
                              "username": username,
                              "password": password
                          }).text)['access_token']
        token = f'Bearer {token}'
        return token

    def send_message(self, sender_id, message):
        response = requests.post(f'{self.url}/api/conversations/{sender_id}/messages',
                                 json={"message": message},
                                 headers={"Authorization": self.token})
        return response.text

    def send_nlu_data(self, text, intent):
        response = requests.post(f'{self.url}/api/projects/default/training_examples',
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

    def list_conversations(self):
        response = requests.get(f'{self.url}/api/conversations',
                                headers={"Authorization": self.token})
        return json.loads(response.text)

    def del_conversation(self, conversation_id):
        response = requests.delete(f'{self.url}/api/conversations/{conversation_id}',
                                   headers={"Authorization": self.token})
        # logging.error(response.ok)
        return response.ok

    def clean_conversations(self):
        conversations = self.list_conversations()
        # logging.info(conversations)
        for conv in conversations:
            self.del_conversation(conv['sender_id'])

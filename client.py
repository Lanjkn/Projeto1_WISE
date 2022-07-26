from flask import Flask
import requests



json_data = {
    'file_loc': 'clean_dataset_customer_churn.csv',
    'y': 'SITUACAO'
}

if __name__ == '__main__':
    requests.post('http://127.0.0.1:5000/set_data/', json=json_data)
    requests.get('http://127.0.0.1:5000/create_model/RFC')

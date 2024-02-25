from flask import Flask, request, jsonify
import psycopg2
from psycopg2 import Error
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

cursor = None
# Connect to your PostgreSQL database
# try:
#     connection = psycopg2.connect(user="default",
#                                   password="fjyBtDhv35lz",
#                                   host="ep-damp-boat-a4li04fz.us-east-1.aws.neon.tech",
#                                   port="5432",
#                                   database="verceldb",
#                                   sslmode="require")

#     cursor = connection.cursor()
#     print("Connected to PostgreSQL successfully")

# except (Exception, Error) as error:
#     print("Error while connecting to PostgreSQL", error)

# # Create users table if not exists
# def create_user_table():
#     create_table_query = '''CREATE TABLE IF NOT EXISTS users
#           (id SERIAL PRIMARY KEY,
#            name VARCHAR(80) NOT NULL,
#            email VARCHAR(80) UNIQUE NOT NULL,
#            password_hash VARCHAR(128) NOT NULL,
#            time FLOAT,
#            change FLOAT);'''
#     cursor.execute(create_table_query)
#     connection.commit()

# create_user_table()







# Endpoint to handle login requests
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    print("data received: ", data)
    
    if 'email' not in data or 'password_hash' not in data:
        print("Email and password are required")
        return jsonify({'error': 'Email and password are required'}), 400

    email = data['email']
    password = data['password_hash']

    try:
        cursor.execute("SELECT * FROM users WHERE email = %s;", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            print("Login successful")
            return jsonify({'message': 'Login successful'}), 200
        else:
            print("Invalid email or password")
            return jsonify({'error': 'Invalid email or password'}), 401

    except (Exception, Error) as error:
        print("Error while fetching data from PostGreSQL", error)
        return jsonify({'error': 'An error occurred'}), 500
    
    
    
    
    
    
# Endpoint to handle signup requests
@app.route('/signup', methods=['POST'])
def signup():
    print("poop")
    print(request)
    data = request.get_json()
    print("data received: ", data)
    if 'name' not in data or 'email' not in data or 'password_hash' not in data:
        print("Name, username, and password are required")
        return jsonify({'error': 'Name, username, and password are required'}), 400

    name = data['name']
    email = data['email']
    password = data['password_hash']

    hashed_password = generate_password_hash(password)

    try:
        cursor.execute("INSERT INTO users (name, email, password_hash, time, change) VALUES (%s, %s, %s, %s, %s);", (name, email, hashed_password, 12, 15))
        # connection.commit()
        print("Signup successful")
        return jsonify({'message': 'Signup successful'}), 200

    except (Exception, Error) as error:
        print("Error while inserting data into PostgreSQL", error)
        return jsonify({'error': 'An error occurred'}), 500
    
    

from adver import attack_img

@app.route('/attack', methods=['POST'])
def attack():
    print("started")
    data = request.get_json()
    print("data received: ", data)
    if 'filename' not in data:
        print("Filename is required")
        return jsonify({'error': 'Filename is required'}), 400

    filename = data['filename']
    epsilon = data.get('epsilon', 0.05)
    lr = data.get('lr', 0.05)

    try:
        attack_img(filename, epsilon, lr)
        print("Attack successful")
        return jsonify({'message': 'Attack successful'}), 200

    except (Exception, Error) as error:
        print("Error while attacking image", error)
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)

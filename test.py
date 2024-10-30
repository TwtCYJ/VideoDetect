from flask import Flask, render_template
from flask_socketio import SocketIO, send

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    print('Received message: ' + msg)
    send('Echo: ' + msg)

if __name__ == '__main__':
    # 在这里设置你想要的端口号，例如 5001
    socketio.run(app, host='127.0.0.1', port=5001)

from app import app, db, socketio

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

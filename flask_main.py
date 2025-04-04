from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
from image_processing import blur_cars_and_persons

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class CapturedPicture(db.Model):
    id= db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    transformed_image_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('pictures', lazy=True))

    def __init__(self, image_path, transformed_image_path):
        self.image_path = image_path
        self.transformed_image_path = transformed_image_path
    def __repr__(self):
        return f'<CapturedPicture {self.image_path}>'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    tkn = db.Column(db.String(120), unique=True, nullable=False)
    image_limit = db.Column(db.Integer, default=4000000)
    image_count = db.Column(db.Integer, default=0)

    def __init__(self, name, email, password, tkn):
        self.name = name
        self.email = email
        self.password = password
        self.tkn = tkn
        self.image_limit = 4000000
        self.image_count = 0

    def __repr__(self):
        return f'<User {self.name}>'

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

# Form with { "picture": <b64encoded_image>, plateRectangleLocation: [x1, y1, x2, y2], plateText: "ABC123" }
@app.route('/process_picture', methods=['POST'])
def process_picture():
    #Check header for token 'Authorization'
    auth = request.headers.get('Authorization')
    if not auth:
        return 'Authorization header missing', 401
    
    # Check if the token is valid
    user = User.query.filter_by(tkn=auth).first()
    if not user:
        return 'Invalid token', 403
    if user.image_count >= user.image_limit:
        return 'Image limit reached', 403
    # Increment the image count
    user.image_count += 1
    db.session.commit()
    # Check json data
    if not request.is_json:
        return 'Invalid JSON data', 400
    data = request.get_json()
    if 'picture' not in data:
        return 'No picture provided', 400
    if 'plateRectangleLocation' not in data:
        return 'No plateRectangleLocation provided', 400
    if 'plateText' not in data:
        return 'No plateText provided', 400
    # Check if the picture is a valid base64 image
    if not data['picture'].startswith('data:image/png;base64,'):
        return 'Invalid picture format', 400
    # Decode the base64 image
    import base64
    import io
    import cv2
    import numpy as np
    picture_data = data['picture'].split(',')[1]
    picture_data = base64.b64decode(picture_data)
    # Convert the byte data to a numpy array
    nparr = np.frombuffer(picture_data, np.uint8)
    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return 'Invalid image data', 400
    # Get the plate rectangle location
    plate_rectangle_location = data['plateRectangleLocation']
    if len(plate_rectangle_location) != 4:
        return 'Invalid plateRectangleLocation format', 400
    x1, y1, x2, y2 = plate_rectangle_location
    # Check if the coordinates are valid
        
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        return 'Invalid plateRectangleLocation coordinates', 400
    # Get the plate text
    plate_text = data['plateText']
    if not isinstance(plate_text, str):
            
        return 'Invalid plateText format', 400
    # Check if the plate text is valid
    if not plate_text.isalnum():
        return 'Invalid plateText format', 400
    # Check if the plate text is empty
    if not plate_text:
        return 'Invalid plateText format', 400
    # Check if the plate text is too long
    if len(plate_text) > 10:
        return 'Invalid plateText format', 400
    # Save the image to a temporary location
    import uuid

    # Save the image to a temporary location with random name
    random_name=  str(uuid.uuid4()) + ".png"
    file_path = f'./input/{random_name}'
    cv2.imwrite(file_path, image)
    # Check if the file was saved successfully
    if not os.path.exists(file_path):
        return 'Failed to save image', 500
    # Process the image using YOLOv8
    # Call the function to blur cars and persons
    blurred_image = blur_cars_and_persons(image, plate_rectangle_location)



    # Process the picture using YOLOv8


    # For demonstration, we'll just simulate a transformed image path
    transformed_image_path = f'./output/transformed_{file.filename}'

    # Save the captured picture information to the database
    new_picture = CapturedPicture(image_path=file_path, transformed_image_path=transformed_image_path)
    db.session.add(new_picture)
    db.session.commit()
    return f'Picture processed and saved at {transformed_image_path}'



from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import qrcode
import os

app = Flask(__name__)

# チェス盤の検出（省略）
def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            return approx
    
    return None

# 駒の検出（テンプレートマッチング）
def detect_pieces(frame, templates):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for piece_name, template in templates.items():
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            cv2.putText(frame, piece_name, (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# テンプレート画像をロードする関数
def load_templates():
    piece_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    templates = {}
    for name in piece_names:
        template = cv2.imread(f"templates/{name}.png", 0)
        if template is not None:
            templates[name] = template
        else:
            print(f"Error: Could not load template for {name}")
    return templates

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            templates = load_templates()
            board_contour = detect_chessboard(img)
            if board_contour is not None:
                detect_pieces(img, templates)
            
            _, img_encoded = cv2.imencode('.jpg', img)
            img_path = os.path.join('static', 'output.jpg')
            cv2.imwrite(img_path, img)
            return send_file(img_path, mimetype='image/jpeg')

    return render_template('upload.html')

@app.route('/qrcode')
def generate_qr():
    ip_address = request.host.split(':')[0]
    url = f'http://{ip_address}:5000'
    qr = qrcode.make(url)
    qr_path = os.path.join('static', 'qrcode.png')
    qr.save(qr_path)
    return render_template('qrcode.html', qr_path=qr_path, url=url)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, request
import os
import time
from predict_calibrated import predict, THRESHOLD

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cloud_cost = 0.0000926

@app.route("/", methods=["GET", "POST"])
def home():

    score = None
    label = None
    image = None
    filename = None
    elapsed = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            filename = file.filename

            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            start = time.perf_counter()
            score = predict(path)
            elapsed = (time.perf_counter() - start) * 1000

            label = "SCREEN" if score >= THRESHOLD else "REAL"

            image = path.replace("\\", "/")

    return render_template(
        "index.html",
        score=score,
        label=label,
        image=image,
        filename=filename,
        elapsed=elapsed,
        cloud_cost=cloud_cost
    )


if __name__ == "__main__":
    app.run(debug=True)
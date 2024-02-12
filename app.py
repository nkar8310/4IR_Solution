from flask import Flask, render_template
from xg import xg_blueprint
from fb import fb_blueprint

app = Flask(__name__)

# Register the blueprints
app.register_blueprint(xg_blueprint)
app.register_blueprint(fb_blueprint)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

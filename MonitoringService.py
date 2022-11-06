from waitress import serve
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    try :
        with open("data/alerts.txt", "r") as f :
            lines = f.readlines()
    except :
        lines = []

    html = "<h1>Alerts</h1>"

    if len(lines) == 0 :
        return html + "<br />No alerts", 200

    html = html + "<table><tr><td><strong>Content</td><td>Action</td></tr>"
    for i, line in enumerate(lines):
        html = html + "<tr><td>" + line + "</td><td><a href='confirm/" + str(i) + "'>legit</a> / <a href='ignore/" + str(i) + "'>ignore</a></td></tr>"
    html = html + "</table>"

    return html, 200

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>Not found.</p>", 404


@app.route('/confirm/<id>', methods=['GET'])
def confirm(id):
    html = "Delete line " + str(id)
    with open("data/alerts.txt", "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i in range(len(d)):
            if str(i) != id:
                f.write(d[i])
        f.truncate()

    html = html + "<meta http-equiv='refresh' content='0;URL=../'>"
    return html, 200

@app.route('/ignore/<id>', methods=['GET'])
def ignore(id):
    html = "Move line " + str(id)
    line_moving = ""
    with open("data/alerts.txt", "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i in range(len(d)):
            if str(i) != id:
                f.write(d[i])
            else:
                line_moving = d[i]
        f.truncate()
    with open("data/standard.txt", "a") as f:
        f.write(line_moving)

    html = html + "<meta http-equiv='refresh' content='0;URL=../'>"
    return html, 200


@app.route('/force_train/', methods=['GET'])
def trainModel():
    html = ""
    return html, 200

if __name__ == "__main__":
    print('Monitoring service starting')
    serve(app, host='0.0.0.0', port=2000)

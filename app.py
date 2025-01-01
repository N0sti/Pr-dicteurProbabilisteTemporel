from flask import Flask, render_template, send_file
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Vérifie si le fichier existe
    file_path = "graphique.png"
    if os.path.exists(file_path):
        return render_template("index.html", graphique=True)
    else:
        return render_template("index.html", graphique=False)

@app.route('/graphique')
def afficher_graphique():
    file_path = "graphique.png"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "<h1>Graphique non disponible. Veuillez le générer d'abord.</h1>"

if __name__ == '__main__':
    app.run(debug=True) 

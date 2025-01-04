from flask import Flask, render_template, send_file
import os
import threading
import time

app = Flask(__name__)

# Variable pour suivre si SolarPredict est en cours d'exécution
solar_predict_running = False

def run_solar_predict():
    global solar_predict_running
    while True:
        if not solar_predict_running:
            try:
                solar_predict_running = True
                os.system("python SolarPredict.py")  # Exécuter tout le fichier SolarPredict
            finally:
                solar_predict_running = False  # Indiquer que SolarPredict a terminé
        time.sleep(300)  # Attendre une heure avant la prochaine exécution

@app.route('/')
def home():
    global solar_predict_running

    # Chemin du fichier du graphique
    file_path = "graphique.png"

    if solar_predict_running:
        # Si SolarPredict est en cours d'exécution
        if os.path.exists(file_path):
            # Si un graphique existe déjà
            return render_template("index.html", graphique=True, en_cours=True)
        else:
            # Si aucun graphique n'existe
            return render_template("index.html", graphique=False, en_cours=True)
    else:
        # Si SolarPredict n'est pas en cours d'exécution
        if os.path.exists(file_path):
            # Si un graphique existe déjà
            return render_template("index.html", graphique=True, en_cours=False)
        else:
            # Si aucun graphique n'existe
            return render_template("index.html", graphique=False, en_cours=False)

@app.route('/graphique')
def afficher_graphique():
    file_path = "graphique.png"
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "<h1>Graphique non disponible. Le code est en cour d'exécution.</h1>"

if __name__ == '__main__':
    # Lancer SolarPredict dans un thread séparé pour s'exécuter périodiquement
    threading.Thread(target=run_solar_predict, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama, LlamaGrammar
import json
import re
import os
import time

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

# --- 1. CONFIGURATION DE LA GRAMMAIRE (Force le format JSON sans bavardage) ---
GBNF_GRAMMAR = r"""
root   ::= object
object ::= "{" space items "}"
items  ::= pair ( "," space pair )*
pair   ::= string ":" space value
string ::= "\"" [^"]* "\""
value  ::= string | number | "null"
number ::= [0-9]+
space  ::= " "?
"""
grammar = LlamaGrammar.from_string(GBNF_GRAMMAR)

# --- 2. CHARGEMENT DU MODÈLE ---
print(f"Chargement du modèle: {MODEL_NAME}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,         # Réduit de 2048 à 1024 pour gagner en vitesse de lecture
    n_threads=16,       # 16 cœurs pour l'inférence (optimal pour tes 24 vCPUs)
    n_batch=512,        # Pour traiter le prompt plus rapidement
    use_mlock=True,     # Verrouille en RAM pour éviter les accès disque
    verbose=False
)

def nettoyer_telephone(numero):
    """Nettoie et normalise un numéro de téléphone français"""
    if not numero: return None
    numero = str(numero).replace(' ', '').replace('.', '').replace('-', '')
    if numero.startswith('+33'): numero = '0' + numero[3:]
    elif numero.startswith('0033'): numero = '0' + numero[4:]
    if len(numero) == 10 and numero.startswith('0'): return numero
    return numero

def structurer_carte_visite(ocr_text, model):
    # --- 3. PROMPT ALLÉGÉ (Plus de 500 tokens économisés ici) ---
    # On enlève les règles et l'exemple long car la grammaire gère la structure.
    prompt = f"""<|im_start|>system
Extract business card data into JSON format. Use null if missing.
Fields: prenom, nom, email, telephone_mobile, telephone_fixe, adresse, code_postal, ville, site_web, societe, fonction, siret.<|im_end|>
<|im_start|>user
{ocr_text}<|im_end|>
<|im_start|>assistant
"""

    print("\n--- DEBUT DU TRAITEMENT ---")
    t_start = time.perf_counter()

    # --- 4. APPEL OPTIMISÉ ---
    output = model(
        prompt,
        max_tokens=200,    # Réduit car un JSON de carte de visite dépasse rarement 100 tokens
        temperature=0,      # 0 est plus rapide et plus précis pour l'extraction
        grammar=grammar,    # LA SOLUTION : On force l'IA à rester dans le JSON
        stop=["<|im_end|>"]
    )

    duration = time.perf_counter() - t_start
    response = output["choices"][0]["text"].strip()
    
    print(f"TEMPS ÉCOULÉ : {duration:.2f} secondes")
    print(f"TOKENS LUS : {output['usage']['prompt_tokens']}")
    print(f"TOKENS GÉNÉRÉS : {output['usage']['completion_tokens']}")
    print("-" * 30)

    try:
        result = json.loads(response)
        # Nettoyage automatique des téléphones après extraction
        for key in ['telephone_mobile', 'telephone_fixe']:
            if result.get(key):
                result[key] = nettoyer_telephone(result[key])
        return result
    except Exception as e:
        print(f"Erreur de parsing : {e}")
        return {"error": "Format JSON invalide"}

@app.route('/api/extract', methods=['POST'])
def extract_business_card():
    try:
        data = request.get_json()
        texte_ocr = data.get('texte_ocr', '')
        if not texte_ocr:
            return jsonify({"error": "texte_ocr manquant"}), 400

        resultat = structurer_carte_visite(texte_ocr, llm)
        return jsonify(resultat), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Serveur prêt sur le port 5000")
    app.run(host='0.0.0.0', port=5000)
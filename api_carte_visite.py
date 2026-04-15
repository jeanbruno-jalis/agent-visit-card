from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import re
import os
import time

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nom du fichier .gguf (le modèle)
MODEL_NAME = "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf"
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

print(f"Serveur démarré - Modèle: {MODEL_NAME}")


def nettoyer_telephone(numero):
    """Nettoie et normalise un numéro de téléphone français"""
    if not numero:
        return None

    numero = str(numero).replace(' ', '').replace('.', '').replace('-', '')

    if numero.startswith('+33'):
        numero = '0' + numero[3:]
    elif numero.startswith('0033'):
        numero = '0' + numero[4:]

    if len(numero) == 10 and numero.startswith('0'):
        return numero

    return numero


def parse_manuellement(ocr_text):
    """Parsing de secours si le JSON échoue"""
    result = {
        "prenom": None, "nom": None, "email": None,
        "telephone_mobile": None, "telephone_fixe": None,
        "adresse": None, "code_postal": None, "ville": None,
        "site_web": None, "societe": None,"siret": None, "fonction": None,
        "_fallback": True
    }

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', ocr_text)
    if email_match:
        result['email'] = email_match.group(0)

    mobile_match = re.search(r'(?:\+33\s?[67]|0[67])[\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{2}', ocr_text)
    if mobile_match:
        result['telephone_mobile'] = nettoyer_telephone(mobile_match.group(0))

    fixe_match = re.search(r'(?:\+33\s?[1-58-9]|0[1-58-9])[\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{2}[\s\.]?\d{2}', ocr_text)
    if fixe_match:
        tel_fixe = nettoyer_telephone(fixe_match.group(0))
        if tel_fixe != result.get('telephone_mobile'):
            result['telephone_fixe'] = tel_fixe

    web_match = re.search(r'(?:www\.)?[\w\-]+\.(?:com|fr|net|org|be|eu)', ocr_text, re.IGNORECASE)
    if web_match:
        result['site_web'] = web_match.group(0)

    cp_match = re.search(r'\b\d{5}\b', ocr_text)
    if cp_match:
        result['code_postal'] = cp_match.group(0)
        
    siret_match = re.search(r'\b(?:\d[\s]*){14}\b', ocr_text)
    if siret_match:
        siret = re.sub(r'\s+', '', siret_match.group(0))
        result['siret'] = siret


    return result


def structurer_carte_visite(ocr_text, model):
    """
    Prend le texte brut OCR d'une carte de visite
    et retourne un dictionnaire structuré
    """
    prompt = f"""You are a JSON extraction assistant. Extract information from this business card and output ONLY valid JSON.

Business card text:
{ocr_text}

IMPORTANT RULES for phone numbers:
- If number starts with +33, replace ONLY the +33 with 0
- If number starts with +33(0), replace ONLY the +33(0) with 0
- If number already starts with 0, keep it as it is (example: 0412345678 stays 0412345678)
- If a number contains spaces and begins with 07, delete spaces and keep it as it is
- else If a number contains spaces and begins with 06, delete spaces and keep it as it is
- Do NOT change any other digits
- Do NOT add 0 if number doesn't have +33
- DO NOT MISSING values of phone numbers

IMPORTANT RULES for postal code:
-The postal code always contains five digits and is always located immediately before the city name (example: 13014 Marseille, postal code is 13014) 

IMPORTANT RULES for SIRET
-The SIRET always contains exactly 14 digits.
-The SIRET contains only numbers (no letters).
-The SIRET may contain spaces between digits, but no other characters.
-The SIRET is often preceded by the word “SIRET” (example: SIRET: 123 456 789 01234).
-The SIRET can appear grouped in blocks (example: 123 456 789 01234) but must be normalized to a 14-digit continuous number (12345678901234).
-If the number does not contain exactly 14 digits after removing spaces, it is NOT a valid SIRET.
-Does not take the Siret value mentioned in the output format
Output format (replace values, use null if missing):
{{"prenom": "John", "nom": "Doe", "email": "john@example.com", "telephone_mobile": "0612345678", "telephone_fixe": "0412345678", "adresse": "123 Main St", "code_postal": "75001", "ville": "Paris", "site_web": "www.example.com", "societe": "ABC Corp", "fonction": "Manager","siret": "82175381100032"}}

JSON:"""


 # ------------------- afficher le nombre de tokens -------------------
    # tokens = model.tokenize(prompt)
    # print(f"[INFO] Nombre de tokens dans le prompt : {len(tokens)}")
    


    print("\n" + "=" * 60)
    print("PROMPT ENVOYÉ AU MODÈLE:")
    print(prompt)
    print("=" * 60 + "\n")

    output = model(
        prompt,
        max_tokens=300,
        temperature=0.05,
        stop=["</s>", "[INST]"]
    )

    response = output["choices"][0]["text"].strip()

    print("\n" + "=" * 60)
    print("RÉPONSE BRUTE DU MODÈLE:")
    print(response)
    print("=" * 60 + "\n")

    # Nettoyage
    response = re.sub(r'^```json\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'^```\s*', '', response)
    response = re.sub(r'```\s*$', '', response)
    response = re.sub(r'</s>.*$', '', response, flags=re.DOTALL)
    response = response.strip()

    json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_match:
        try:
            result = json.loads(json_match.group(0))

            if result.get('telephone_mobile'):
                result['telephone_mobile'] = nettoyer_telephone(result['telephone_mobile'])

            if result.get('telephone_fixe'):
                result['telephone_fixe'] = nettoyer_telephone(result['telephone_fixe'])

            print("JSON PARSÉ AVEC SUCCÈS")
            return result

        except json.JSONDecodeError:
            print("ERREUR JSON → fallback manuel")
            return parse_manuellement(ocr_text)

    print("AUCUN JSON TROUVÉ → fallback manuel")
    return parse_manuellement(ocr_text)


# Chargement du modèle UNE SEULE FOIS
print("Chargement du modèle en mémoire...")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=35,
    n_ctx=2048,
    verbose=False
)

print("Modèle chargé")


@app.route('/api/extract', methods=['POST'])
def extract_business_card():
    start_time = time.time()

    try:
        data = request.get_json()
        texte_ocr = data.get('texte_ocr', '')

        if not texte_ocr:
            return jsonify({"error": "texte_ocr manquant"}), 400

        resultat = structurer_carte_visite(texte_ocr, llm)

        elapsed = time.time() - start_time
        print(f"TEMPS: {elapsed:.2f}s")

        return jsonify(resultat), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#Vérifier que le service fonctionne
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK", "model": MODEL_NAME}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)






import streamlit_authenticator as stauth
import sys

# --- CAMBIA ESTAS CONTRASEÑAS ---
passwords_to_hash = ['admin123', 'medico456']
# --------------------------------

hashed_passwords = stauth.Hasher(passwords_to_hash).generate()

print("--- Copia y pega estos hashes en tu config.yaml ---")
print(hashed_passwords)
print("-----------------------------------------------------")
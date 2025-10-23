import streamlit_authenticator as stauth
import sys

# --- CAMBIA ESTAS CONTRASEÑAS ---
passwords_to_hash = ['admin123', 'medico456']
# --------------------------------

# Creamos una instancia del Hasher
hasher = stauth.Hasher()

# Usamos un bucle (list comprehension) para hashear CADA contraseña
# El método .hash() toma una contraseña (string), no una lista.
hashed_passwords = [hasher.hash(p) for p in passwords_to_hash]

print("--- Copia y pega estos hashes en tu config.yaml ---")
print(hashed_passwords)
print("-----------------------------------------------------")
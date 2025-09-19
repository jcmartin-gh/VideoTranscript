# simple_login.py
from __future__ import annotations
import os
import hmac
import streamlit as st

def _expected_password(key: str) -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return str(os.environ.get(key, ""))

def require_login(
    app_name: str = "App",
    password_key: str = "APP_PASSWORD",
    session_flag: str = "is_auth",
) -> bool:
    """
    Login por contraseña única.
    - Define la contraseña en st.secrets['APP_PASSWORD'] o variable de entorno APP_PASSWORD.
    - Mantiene sesión en st.session_state[session_flag].
    """
    expected = _expected_password(password_key)

    # Estado inicial seguro (antes de crear widgets)
    st.session_state.setdefault(session_flag, False)
    st.session_state.setdefault("login_pwd", "")
    st.session_state.setdefault("login_error", "")

    # Callbacks seguros para Streamlit
    def _login_cb():
        pwd = st.session_state.get("login_pwd", "")
        if expected and hmac.compare_digest(pwd, expected):
            st.session_state[session_flag] = True
            st.session_state["login_error"] = ""
            st.session_state["login_pwd"] = ""  # permitido dentro del callback
        else:
            st.session_state[session_flag] = False
            st.session_state["login_error"] = "❌ Contraseña incorrecta."

    def _logout_cb():
        st.session_state[session_flag] = False
        st.session_state["login_pwd"] = ""
        st.session_state["login_error"] = ""

    with st.sidebar:
        st.subheader("User Autentication")
        if not expected:
            st.info(
                f"Configura la contraseña en st.secrets['{password_key}'] "
                f"o en la variable de entorno {password_key}."
            )

        # Widget de contraseña (clave sin guiones bajos dobles)
        st.text_input("Contraseña", type="password", key="login_pwd")

        c1, c2 = st.columns(2)
        c1.button("Entrar", use_container_width=True, on_click=_login_cb)
        if st.session_state[session_flag]:
            c2.button("Salir", use_container_width=True, on_click=_logout_cb)

        # Mensajes
        if st.session_state["login_error"]:
            st.error(st.session_state["login_error"])
        elif st.session_state[session_flag]:
            st.success("✅ Acceso concedido.")

    if not st.session_state[session_flag]:
        st.write("🔒 La aplicación está protegida. Introduce la contraseña para continuar.")
        return False
    return True

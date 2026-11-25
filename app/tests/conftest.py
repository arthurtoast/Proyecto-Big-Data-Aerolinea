"""Las pruebas corren SIN red: cualquier intento de abrir una conexion falla con OSError,
igual que una falla de red real (requests la convierte en ConnectionError)."""
import socket

import pytest


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    def bloqueado(*args, **kwargs):
        raise OSError("las pruebas no deben usar la red")

    monkeypatch.setattr(socket.socket, "connect", bloqueado)
    monkeypatch.setattr(socket, "create_connection", bloqueado)

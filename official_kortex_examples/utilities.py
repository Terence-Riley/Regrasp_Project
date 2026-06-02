"""Small official-style Kortex connection helper.

This mirrors the structure of Kinova's Python examples: parse connection
arguments, create TCP transport/router/session, then expose a context manager.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc


def patch_python310_protobuf_compat():
    """Allow older Kortex wheels with protobuf 3.5.x to import on Python 3.10."""
    for name in ["Mapping", "MutableMapping", "Sequence", "MutableSequence", "Iterable"]:
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


patch_python310_protobuf_compat()

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions  # noqa: E402
from kortex_api.SessionManager import SessionManager  # noqa: E402
from kortex_api.TCPTransport import TCPTransport  # noqa: E402
from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient  # noqa: E402
from kortex_api.autogen.messages import Session_pb2  # noqa: E402


def parse_connection_arguments(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ip", type=str, default="192.168.1.10")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--password", type=str, default="admin")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--transport", choices=["tcp", "mqtt"], default="tcp")
    parser.add_argument("--timeout-ms", type=int, default=20000)
    return parser


class DeviceConnection:
    TCP_PORT = 10000
    MQTT_PORT = 1883

    def __init__(self, ip: str, port: int, username: str, password: str, transport_type: str, timeout_ms: int):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.transport_type = transport_type
        self.timeout_ms = timeout_ms
        if transport_type == "mqtt":
            try:
                from kortex_api.MQTTTransport import MqttTransport
            except ImportError as exc:
                raise RuntimeError("MQTT transport requires paho-mqtt: python -m pip install paho-mqtt") from exc
            self.transport = MqttTransport()
        else:
            self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.session = None

    @staticmethod
    def create_tcp_connection(args):
        transport = getattr(args, "transport", "tcp")
        port = getattr(args, "port", DeviceConnection.TCP_PORT)
        timeout_ms = getattr(args, "timeout_ms", 20000)
        if transport == "mqtt" and port == DeviceConnection.TCP_PORT:
            port = DeviceConnection.MQTT_PORT
        return DeviceConnection(args.ip, port, args.username, args.password, transport, timeout_ms)

    @staticmethod
    def createTcpConnection(args):
        return DeviceConnection.create_tcp_connection(args)

    def __enter__(self):
        print(f"Connecting to {self.ip}:{self.port} with {self.transport_type.upper()}...")
        self.transport.connect(self.ip, self.port)
        print("Transport connected. Creating session...")

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000
        router_options = RouterClientSendOptions()
        router_options.timeout_ms = self.timeout_ms

        if self.transport_type == "mqtt":
            self.session = SessionClient(self.router)
        else:
            self.session = SessionManager(self.router)
        self.session.CreateSession(session_info, router_options)
        print("Session created.")
        return self.router

    def __exit__(self, exc_type, exc_value, traceback):
        if self.session is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = min(self.timeout_ms, 5000)
            self.session.CloseSession(router_options)
        self.transport.disconnect()

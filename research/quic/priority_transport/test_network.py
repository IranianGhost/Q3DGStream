import asyncio
from pathlib import Path
import subprocess
import pytest
from aioquic.asyncio import connect, serve
from aioquic.quic.configuration import QuicConfiguration
from research.quic.priority_transport.transport import (
    ALPN,
    FileJob,
    Receiver,
    send_files,
    send_whole_files,
)


@pytest.fixture(scope="module")
def tls(tmp_path_factory):
    root = tmp_path_factory.mktemp("tls")
    cert = root / "cert.pem"
    key = root / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.mark.parametrize("mode", ["chunked", "whole"])
def test_actual_quic_file_commit_callback(tmp_path, tls, mode):
    async def run():
        cert, key = tls
        notifications = []
        rx = Receiver(tmp_path / "cache", on_file=notifications.append)
        conf = QuicConfiguration(is_client=False, alpn_protocols=ALPN)
        conf.load_cert_chain(cert, key)
        server = await serve(
            "127.0.0.1", 0, configuration=conf, stream_handler=rx.handler
        )
        client = QuicConfiguration(
            is_client=True, alpn_protocols=ALPN, server_name="localhost"
        )
        client.load_verify_locations(cert)
        source = tmp_path / "source"
        source.write_bytes(bytes(range(256)) * 100)
        try:
            async with connect(
                "127.0.0.1",
                server._transport.get_extra_info("sockname")[1],
                configuration=client,
            ) as connection:
                sender = send_files if mode == "chunked" else send_whole_files
                await sender(connection, [FileJob(source, "NTCs/NTC_000000.pth", 0)])
            assert notifications == ["NTCs/NTC_000000.pth"]
            assert (
                tmp_path / "cache" / notifications[0]
            ).read_bytes() == source.read_bytes()
            assert not rx.errors
        finally:
            server.close()
            await rx.close()

    asyncio.run(run())


def test_receiver_reports_timeout_type(tmp_path, monkeypatch):
    import json

    class Writer:
        payload = b""

        def write(self, data):
            self.payload += data

        def write_eof(self):
            pass

    async def expired(awaitable, timeout):
        awaitable.close()
        raise TimeoutError()

    async def check():
        receiver = Receiver(tmp_path)
        writer = Writer()
        monkeypatch.setattr(asyncio, "wait_for", expired)
        await receiver.receive(asyncio.StreamReader(), writer)
        assert json.loads(writer.payload)["error"] == "TimeoutError()"
        await receiver.close()

    asyncio.run(check())

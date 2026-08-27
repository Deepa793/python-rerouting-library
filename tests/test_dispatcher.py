import pytest

from python_rerouting_library.dispatcher import Dispatcher
from python_rerouting_library.exceptions import (
    CloudBackendError,
    DispatchError,
    LocalBackendError,
)
from python_rerouting_library.router import RouteDecision


class FakeRouter:
    def __init__(self, label):
        self.label = label

    def route(self, query):
        probabilities = {
            "simple": 0.1,
            "uncertain": 0.5,
            "complex": 0.9,
        }

        return RouteDecision(
            label=self.label,
            confidence=0.9 if self.label != "uncertain" else 0.0,
            complex_probability=probabilities[self.label],
            latency_ms=1.0,
        )


class FakeBackend:
    def __init__(self, name):
        self.name = name

    def generate(self, query):
        return f"{self.name}: {query}"


class FailingLocalBackend:
    name = "local"

    def generate(self, query):
        raise LocalBackendError("Local failed")


class FailingCloudBackend:
    name = "cloud"

    def generate(self, query):
        raise CloudBackendError("Cloud failed")


def test_simple_goes_local():
    dispatcher = Dispatcher(
        router=FakeRouter("simple"),
        simple_backend=FakeBackend("local"),
        complex_backend=FakeBackend("cloud"),
    )

    result = dispatcher.run("hello")

    assert result.backend_name == "local"
    assert result.route.label == "simple"
    assert result.fallback_used is False


def test_complex_goes_cloud():
    dispatcher = Dispatcher(
        router=FakeRouter("complex"),
        simple_backend=FakeBackend("local"),
        complex_backend=FakeBackend("cloud"),
    )

    result = dispatcher.run("design architecture")

    assert result.backend_name == "cloud"
    assert result.route.label == "complex"
    assert result.fallback_used is False


def test_uncertain_goes_cloud():
    dispatcher = Dispatcher(
        router=FakeRouter("uncertain"),
        simple_backend=FakeBackend("local"),
        complex_backend=FakeBackend("cloud"),
    )

    result = dispatcher.run("ambiguous request")

    assert result.backend_name == "cloud"
    assert result.route.label == "uncertain"
    assert result.fallback_used is False


def test_local_failure_falls_back_to_cloud():
    dispatcher = Dispatcher(
        router=FakeRouter("simple"),
        simple_backend=FailingLocalBackend(),
        complex_backend=FakeBackend("cloud"),
    )

    result = dispatcher.run("hello")

    assert result.backend_name == "cloud"
    assert result.route.label == "simple"
    assert result.fallback_used is True


def test_local_and_cloud_failure_raises_dispatch_error():
    dispatcher = Dispatcher(
        router=FakeRouter("simple"),
        simple_backend=FailingLocalBackend(),
        complex_backend=FailingCloudBackend(),
    )

    with pytest.raises(DispatchError):
        dispatcher.run("hello")


def test_complex_cloud_failure_raises_dispatch_error():
    dispatcher = Dispatcher(
        router=FakeRouter("complex"),
        simple_backend=FakeBackend("local"),
        complex_backend=FailingCloudBackend(),
    )

    with pytest.raises(DispatchError):
        dispatcher.run("design an architecture")


def test_uncertain_cloud_failure_raises_dispatch_error():
    dispatcher = Dispatcher(
        router=FakeRouter("uncertain"),
        simple_backend=FakeBackend("local"),
        complex_backend=FailingCloudBackend(),
    )

    with pytest.raises(DispatchError):
        dispatcher.run("ambiguous request")
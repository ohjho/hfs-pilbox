"""Tests for app.py wiring: per-tab concurrency limits and their env-var overrides."""

import importlib

import app


def test_tiered_concurrency_limits():
    for iface in (app.annotate_interface, app.crop_interface, app.mask_interface):
        assert iface.concurrency_limit == app.IMAGE_CONCURRENCY
    for iface in (
        app.annotate_video_interface,
        app.crop_video_interface,
        app.mask_video_interface,
    ):
        assert iface.concurrency_limit == app.VIDEO_CONCURRENCY


def test_tiering_invariant():
    assert app.IMAGE_CONCURRENCY >= app.VIDEO_CONCURRENCY >= 1
    assert app.QUEUE_MAX_SIZE >= 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("PILBOX_IMAGE_CONCURRENCY", "5")
    monkeypatch.setenv("PILBOX_VIDEO_CONCURRENCY", "3")
    monkeypatch.setenv("PILBOX_QUEUE_MAX_SIZE", "7")
    try:
        reloaded = importlib.reload(app)
        assert reloaded.IMAGE_CONCURRENCY == 5
        assert reloaded.VIDEO_CONCURRENCY == 3
        assert reloaded.QUEUE_MAX_SIZE == 7
        assert reloaded.crop_interface.concurrency_limit == 5
        assert reloaded.mask_video_interface.concurrency_limit == 3
    finally:
        monkeypatch.undo()
        importlib.reload(app)

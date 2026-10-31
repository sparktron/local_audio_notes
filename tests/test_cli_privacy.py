from local_audio_notes.cli import main as cli
from local_audio_notes.errors import NotesError


def test_library_prints_do_not_escape_cli(monkeypatch, capsys):
    def private_failure(*args):
        print("PRIVATE TRANSCRIPT EXCERPT")
        raise NotesError("Safe stage error")

    monkeypatch.setattr(cli, "process", private_failure)
    monkeypatch.setattr(cli, "enable_runtime", lambda **kwargs: None)
    assert cli.main(["run", "unused-in-this-mock"]) == 1
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.out + captured.err
    assert "Safe stage error" in captured.err

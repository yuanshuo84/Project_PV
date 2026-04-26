# Project PV

Project PV is a small Python desktop app for defining skeleton motion and
exporting the result as a pixel GIF animation.

Everything is experimental with AI generated code. trying to improve the code quality by add more skills.

If you like it, i do not mind any tips!

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=JSTCTTLDQSCSE)


## Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verify

```powershell
python -m pytest
```

## Run

```powershell
python -m project_pv.main
```

After installation, the UI can also be started with:

```powershell
project-pv
```

The left side of the UI controls the skeleton figure, keyframed joint
positions, and motion. The right side shows the rasterized frame preview and
exports the animation as a `.gif`.
Exported GIFs use a transparent background.

## Logging

Project PV writes runtime logs to `logs/project_pv.log`. To capture detailed
debug logs while running the app:

```powershell
$env:PROJECT_PV_LOG_LEVEL = "DEBUG"
python -m project_pv.main
```

## VS Code

This repository includes VS Code configuration in `.vscode/`:

- `Project PV: Run UI` launches the desktop app.
- `Project PV: Tests` debugs the pytest suite.
- Tasks are available for `Install dev`, `Run UI`, and `Run tests`.

# Records docs/demo.gif from a real run of the agent.
# Run from the repository root:  vhs docs/demo.tape
# Requirements: vhs (https://github.com/charmbracelet/vhs), ttyd, ffmpeg, and a working .env
#
# Before pushing the GIF, watch it once: make sure no key, e-mail or private path is visible.
# Increase the final Sleep if the run takes longer on your machine.

Output docs/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 750
Set Padding 20
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

Type "uv run python -m src.runner --only 1,3 --preview 160"
Sleep 600ms
Enter

# Question 1 uses the Wikipedia tool with a historical revision; question 3 needs no tool.
Sleep 40s

lint:
	flake8

pyinstaller:
	cd specfiles && pyinstaller --noconfirm Mint.spec ../scripts/Mint.py

doc:
	mkdocs build && mkdocs gh-deploy

devel:
	Mint --debug --no-browser

serve:
	Mint --no-browser

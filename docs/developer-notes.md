# Developer Notes

## Building and Uploading the Package

To build the source distribution and wheel for `ms-mint-app2`, follow these steps:

1. Navigate to the root directory of the project.
2. Run the following command to create the source distribution (`sdist`) and wheel (`wheel`):

```bash
python -m build
```

   This will generate distribution archives in the `dist` directory.

3. To upload the built package, use `twine`. Ensure you have `twine` installed (`pip install twine` if not). Then, run:

```bash
python -m twine upload dist/*
```

   If you use a custom repository, configure it in `~/.pypirc` and pass `--repository <name>`.

## Windows Executables

To create Windows executables for the `ms-mint-app2` application, use `pyinstaller`. Follow the full guide in `pyinstaller/BUILD_GUIDE.md`. The short version is:

```bash
cd pyinstaller
python create_asari_env.py
python prebuild_matplotlib_cache.py
pyinstaller Mint.spec
```

This will generate a standalone executable based on `pyinstaller/Mint.spec`.

## Documentation Deployment

To build and deploy the documentation using `mkdocs`, follow these steps:

1. Ensure you have `mkdocs` and the Material theme installed (`pip install mkdocs mkdocs-material` if not), or use `pip install -r requirements-dev.txt`.
2. Run the following commands to build the documentation and deploy it to GitHub Pages:

```bash
mkdocs build && mkdocs gh-deploy
```

   The `mkdocs build` command generates the static site in the `site` directory, and `mkdocs gh-deploy` pushes it to the `gh-pages` branch of your GitHub repository.

## Example NGINX Configuration

To run `ms-mint-app2` on a remote server, you need to set up a reverse proxy using NGINX. Here is an example configuration:

    server {
        ...
        location / {
            proxy_pass              http://localhost:9999;
            client_max_body_size    100G;
            proxy_set_header        X-Forwarded-Proto https;
            proxy_set_header        Host $host;
        }
    }

Explanation:

  - `proxy_pass http://localhost:9999;`: Forwards all requests to the MINT application running on port 9999 (default).
  - `client_max_body_size 100G;`: Increases the maximum allowed size of client request bodies to 100GB.
  - `proxy_set_header X-Forwarded-Proto https;`: Sets the `X-Forwarded-Proto` header to `https`.
  - `proxy_set_header Host $host;`: Ensures the `Host` header from the original request is passed to the proxied server.

Then start MINT with `Mint --host 0.0.0.0 --port 9999` (or your chosen port).

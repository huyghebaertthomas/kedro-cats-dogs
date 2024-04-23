# Cats and dogs classification pipeline with Kedro

1. Make a new virtual environment: `python -m venv .venv`
2. Created a .gitignore file containing `.venv`
3. Activate the environment: `.venv\Scripts\activate`
4. Install Kedro and some plugins: `pip install kedro kedro-docker kedro-viz`
5. Verify installation: `kedro info`
6. Create project: `kedro new --name=Cats-Dogs --tools=viz --example=n`
7. Change context: `cd cats-dogs`
8. Create pipeline: `kedro pipeline create classification`
9. Fill in the data catalog

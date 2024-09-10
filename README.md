
```sh
python3 -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
deactivate
pip freeze > requirements.txt
$env:OPENAI_API_KEY=""
docker-compose up -d
```
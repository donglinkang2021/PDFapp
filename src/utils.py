from ollama import Client

def list_model(host: str) -> list[str]:
    client = Client(host)
    response = client.list()
    return [model.model for model in response.models]

if __name__ == '__main__':
    models = list_model('http://localhost:11434')
    print(models)

# python -m src.utils
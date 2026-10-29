import os
import json
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi

def fazer_upload_dataset():
    # Configurações
    usuario_kaggle = os.getenv("KAGGLE_USERNAME")

    if not usuario_kaggle:
        raise ValueError("KAGGLE_USERNAME não encontrado no .env")

    nome_slug_dataset = "hollow-knight-dataset"
    titulo_dataset = "Hollow Knight Hornet Battle Dataset"

    # Caminho da pasta do dataset
    pasta_do_script = os.path.dirname(os.path.abspath(__file__))
    pasta_local_datasets = os.path.join(pasta_do_script, "hornet_boss")

    if not os.path.isdir(pasta_local_datasets):
        raise FileNotFoundError(
            f"Pasta não encontrada: {pasta_local_datasets}"
        )

    # Cria metadata
    metadata = {
        "title": titulo_dataset,
        "id": f"{usuario_kaggle}/{nome_slug_dataset}",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }

    caminho_metadata = os.path.join(
        pasta_local_datasets,
        "dataset-metadata.json"
       )

    with open(caminho_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"📝 Metadata criada: {caminho_metadata}")

    # Autentica
    api = KaggleApi()
    api.authenticate()

    print("✅ Autenticado no Kaggle")

    try:
        print("🆕 Tentando criar o dataset...")

        api.dataset_create_version(
                folder=pasta_local_datasets,
                version_notes="Atualização automática via script",
                dir_mode="zip"
            )

        print("✅ Dataset criado com sucesso!")

    except Exception as e:
        erro = str(e)

        print(f"⚠️ Erro ao criar dataset: {erro}")

        # Se já existir, tenta enviar nova versão
        if (
            "already exists" in erro.lower()
            or "409" in erro
            or "conflict" in erro.lower()
        ):
            print("🔄 Dataset já existe. Criando nova versão...")

            api.dataset_create_version(
                folder=pasta_local_datasets,
                version_notes="Atualização automática via script",
                dir_mode="zip"
            )

            print("✅ Nova versão enviada com sucesso!")

        else:
            raise


if __name__ == "__main__":

    import os
    from dotenv import load_dotenv
    from kaggle.api.kaggle_api_extended import KaggleApi

    load_dotenv()

    print("USERNAME:", os.getenv("KAGGLE_USERNAME"))
    print("TOKEN:", os.getenv("KAGGLE_API_TOKEN")[:10])

    api = KaggleApi()
    api.authenticate()

    print("Autenticado!")

    fazer_upload_dataset()
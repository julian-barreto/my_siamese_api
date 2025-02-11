FROM python:3.10-slim

WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto de la API
EXPOSE 5000

# Comando para correr la API
CMD ["python", "app.py"]

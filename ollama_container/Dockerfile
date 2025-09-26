FROM ollama/ollama
COPY entrypoint.sh /entrypoint.sh
COPY Modelfile Modelfile
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

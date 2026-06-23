v3_config = {
    "url": "url for deepseek_v3",
    "model": "deepseek_chat",
    "api_key": "api_key"
}

sql_service_url = 'url for sql service'

# POC: gateway backbone. The runner overwrites this with the live gateway url/model and
# a fresh bearer token (poc_eval.common.llm_gateway), which get_chat_result passes into
# OpenAI(api_key=..., base_url=...). Exactly the repo's call path, just pointed at the gateway.
gateway_config = {
    "url": "set-by-runner",
    "model": "set-by-runner",
    "api_key": "set-by-runner",
}

config_mapping = {
    "v3": v3_config,
    "gateway": gateway_config,
}
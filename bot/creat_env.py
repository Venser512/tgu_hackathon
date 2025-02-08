# Создаем .env файл с токеном
bot_token = "ВАШ_ТОКЕН_ЗДЕСЬ"  # Замените ВАШ_ТОКЕН_ЗДЕСЬ на ваш реальный токен

with open('.env', 'w') as f:
    f.write(f'BOT_TOKEN={bot_token}\n')

print('.env файл успешно создан с токеном.')
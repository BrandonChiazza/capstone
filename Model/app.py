print("Hello World")

from flask import Flask
from Resources.blueprints import imageAccess
from flask_swagger_ui import get_swaggerui_blueprint

UPLOAD_FOLDER = 'static/uploads/'

application = Flask(__name__)
application.register_blueprint(imageAccess)

SWAGGER_URL = '/api/swagger'  
API_URL = '/static/swagger.yml' 

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)
application.secret_key = "seraudhlajfdoij"  
application.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
if __name__ == "__main__":
    application.debug = True
    application.run(host='0.0.0.0')
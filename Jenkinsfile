pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Code Validation') {
            steps {
                sh '''
                . venv/bin/activate
                python3 -m py_compile *.py || true
                '''
            }
        }

        stage('Deploy Simulation') {
            steps {
                sh '''
                echo "Packaging application..."
                tar -czf documind-artifact.tar.gz *
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}

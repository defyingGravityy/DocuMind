node('agent') {

    try {

        stage('Install Dependencies') {
            sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
            '''
        }

        stage('Code Validation') {
            sh '''
                . venv/bin/activate
                python3 -m py_compile *.py
                echo "Validation successful"
            '''
        }

        stage('Deploy Simulation') {
            sh '''
                echo "Packaging application..."
                tar -czf documind-artifact.tar.gz *
                echo "Deployment artifact created"
            '''
        }

        echo "Pipeline executed successfully!"

    } catch (Exception e) {

        echo "Pipeline failed!"
        throw e
    }
}


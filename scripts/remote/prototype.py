import paramiko
import subprocess

def get_2fa_code():
    code = subprocess.check_output(['python', '/Users/yitan/.gen_otp_mini.py']).decode('utf-8').strip()
    return code

def handler(title, instructions, prompt_list):
    responses = []
    for prompt in prompt_list:
        if "password" in prompt[0].lower():
            responses.append('QQmima13654543@')  # Replace with your actual password
        else:
            responses.append(get_2fa_code())  # Prompt the user to input the 2FA code
    return responses

def ssh_connect():
    paramiko.common.logging.basicConfig(level=paramiko.common.DEBUG)
    t=paramiko.Transport(('login.rc.fas.harvard.edu', 22))
    t.start_client()
    t.auth_interactive_dumb('ytan', handler)
    session = t.open_channel(kind='session')

    # save file
    # session.exec_command('ls > test.txt')
    # # download the file
    # sftp = paramiko.SFTPClient.from_transport(t)
    # sftp.get('test.txt', 'test.txt')
    # sftp.close()
    # session.close()
    # t.close()
    
    session.exec_command('echo "Hello from server"; ls; echo "End of command"')
    # Explicitly wait for the command to complete
    session.recv_exit_status()
    # Reading output
    stdout = session.makefile('r', -1)
    stderr = session.makefile_stderr('r', -1)
    output = stdout.read().decode('utf-8').strip()
    error = stderr.read().decode('utf-8').strip()

    # Print outputs
    if output:
        print('STDOUT:', output)
    if error:
        print('STDERR:', error)

    session.close()
    t.close()

ssh_connect()
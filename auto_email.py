from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.base import MIMEBase
from smtplib import SMTP_SSL
from email import encoders
import os

#qq邮箱smtp服务器
def send_email(receiver, title="AbCFold", message="Result of AbCFold", file_path=""):
    #qq邮箱的stmp服务器
    host_server = 'smtp.qq.com'
    #pwd为qq邮箱授权码，类似于密码
    pwd = "yfiedhsviahabjgh"
    sender_email = "122474603@qq.com"
    
    """这一段是构建smtp服务器"""
    smtp = SMTP_SSL(host=host_server)
    #set_debuglevel()是用来调试的。参数值为1表示开启调试模式，参数值为0关闭调试模式
    smtp.ehlo(host_server)
    smtp.login(sender_email, pwd)

    msg = MIMEMultipart()
    msg["Subject"] = Header(title, "utf-8")
    msg["From"] = "AbCFold <122474603@qq.com>"
    msg["to"] = receiver

    # 添加邮件正文
    msg.attach(MIMEText(message, "plain", "utf-8"))

    # 添加附件
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            mime = MIMEBase('application', 'octet-stream', filename=file_path)
            mime.add_header('Content-Disposition', 'attachment', filename=file_path)
            mime.add_header('Content-ID', '<0>')
            mime.add_header('X-Attachment-Id', '0')
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            msg.attach(mime)
    
    smtp.sendmail(sender_email, receiver, msg.as_string())#如果receiver传入的是一个列表，则可向列表内所有地址发送邮件
    smtp.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send email to receiver")
    parser.add_argument("-r", "--receiver", required=True, help="Receiver email address")
    parser.add_argument("-t", "--title", default="AbCFold", help="Email title")
    parser.add_argument("-m", "--message", default="Result of AbCFold", help="Email message")
    parser.add_argument("-f", "--file_path", default="", help="Attachment file path")
    args = parser.parse_args()
    
    send_email(args.receiver, args.title, args.message, args.file_path)
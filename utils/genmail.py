def generate_mail(query, st_year, en_year, pubs, top_n):
    mail_content = f"""\
            <html>

            <head>
                <div style="font-weight: 1000; color: white;"><u>Rainfall Prediction Result</u></div>
            </head>
            
            <body style="background-color: black; text-align: center;">
                <br> <a href="#">
                <img src="https://i.imgur.com/PiZMOCp.png" width="36%" height="56%"
                    style="vertical-align:middle; align-items: center;"> </a>
                <p style="color: green;">
                    <b>
                        Hello, this is an automated message from DocRel!
                    </b>
                </p>
                <p>
                <div style="color: white">
                    According to the entered data: <br>
                </div>
                <div style="color: goldenrod; font-weight: 520;">
                    Query: {query}. <br>
                    Time frame: {st_year} - {en_year}. <br>
                    Publications per year: {pubs}. <br>
                    Results needed: {top_n}. <br>
                </div>
                </p>
                <p style="color: red; font-weight: 750;">
                    The top publications are: {query}WTVR mm.
                </p>
                <p style="color: white;">
                    (Feel free to leave us feedback on our
                    <a href="mailto:archmages.neural@gmail.com?
                                &cc=
                                &bcc=
                                &subject=Feedback for Rainfall Prediction Page
                                &body=Add whatever suggestions or message you would like to send here" 
                                target="_blank" style="color: white;">
                        Mail
                    </a> or
                    <a href="https://github.com/RpM-Kinshuk/DocRel" target="_blank"> Github Page</a>!)
                </p>
                <p>
                <div style="color: limegreen;">
                    Thank you for using our service!
                </div>
                <div style="color: white; font-weight: 750;"> <br>
                    <div style="color:white;">Regards,</div>
                    <a href="https://www.linkedin.com/in/kinshuk-goel/" target="_blank" style="color: rgb(13, 13, 191);">Kinshuk</a> 
                    <br> 
                    <div style="color:red">Neural Archmages</div>
                </div>
                </p>
            </body>
            
            </html>
        """  # .format(month, day, temp, sphum, relhum, rainfall)
    return mail_content

def generate_mail(query, st_year, en_year, pubs, top_n, results):
    mail_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }}
                h2 {{
                    text-align: center;
                    color: #0056b3;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                table, th, td {{
                    border: 1px solid #ddd;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    color: #333;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    font-size: 14px;
                    color: #666;
                }}
                a {{
                    color: #0056b3;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Semantic Analysis Results</h2>
                <p><strong>Query:</strong> {query}</p>
                <p><strong>Time frame:</strong> {st_year} - {en_year}</p>
                <p><strong>Publications per year:</strong> {pubs}</p>
                <p><strong>Top {top_n} results:</strong></p>
                <table>
                    <thead>
                        <tr>
                            <th>Title</th>
                            <th>Authors</th>
                            <th>Abstract</th>
                            <th>DOI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{res['Title']}</td>
                            <td>{res['Authors']}</td>
                            <td>{res['abstract'][:100]}...</td>
                            <td><a href="{res['DOI']}" target="_blank">Link</a></td>
                        </tr>''' for res in results])}
                    </tbody>
                </table>
                <div class="footer">
                    <p>Thank you for using our service!</p>
                    <p>
                        Regards, 
                        <a href="https://www.linkedin.com/in/kinshuk-goel/" target="_blank">Kinshuk Goel,</a>
                        <a href="https://www.linkedin.com/in/avni-verma-975291230/" target="_blank">Avni Verma,</a>
                        <a href="https://www.linkedin.com/in/priyanshitiwari02/" target="_blank">Priyanshi Tiwari</a> and
                        <a href="https://www.linkedin.com/in/anubhav-bhattacharyya-26b1bb24a/" target="_blank">Anubhav Bhattacharyya</a>
                        <br>Neural Archmages
                    </p>
                    <p><a href="mailto:archmages.neural@gmail.com">Mail us</a> or visit our <a href="https://github.com/RpM-Kinshuk/DocRel" target="_blank">GitHub page</a>.</p>
                </div>
            </div>
        </body>
        </html>
    """
    return mail_content

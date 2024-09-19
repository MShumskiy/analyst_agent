
from accelerate import Accelerator
from transformers import AutoTokenizer,AutoModelForCausalLM


import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df_name = 'sales_and_price_data'
def simple_linear_regression(df, x_col, y_col):
    # Extract x and y values from the DataFrame
    x = df[[x_col]].values.reshape(-1, 1)  # Reshape for sklearn which expects a 2D array
    y = df[y_col].values
    
    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Get the slope (m) and intercept (b)
    m = model.coef_[0]
    b = model.intercept_
    r_squared = model.score(x, y)
    
    # Return the equation of the line
    return m, b,r_squared,model

def lin_reg_predict(model,df,x_col,y_col):
    y_hat = model.predict(df[x_col].values.reshape(-1, 1))
    df_hat = df.copy()
    pred_col = y_col + '_hat'
    df_hat[pred_col] = y_hat
    return df_hat

def plot_results_pred(df,x_col,y_col,save_image_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], color='blue', label=y_col, s = 5)
    plt.plot(df[x_col], df[y_col + '_hat'], color='red', label=y_col+ ' predicted')

    # Adding labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(y_col + ' vs ' + y_col + ' predicted')
    plt.legend()

    plt.savefig(save_image_path)

    # Show the plot
    plt.show()

def lin_reg(df_name,x_col,y_col):
    """
    Perform a simple linear regression on a dataset and optionally plot the results.

    Parameters:
    df_name (str): The name of the dataframe to use.
    x_col (str): The name of the column to be used as the independent variable (X) in the regression.
    y_col (str): The name of the column to be used as the dependent variable (Y) in the regression.
    plot (bool): A flag indicating whether to plot the results (True) or not (False).

    Returns:
    tuple: A tuple containing the slope (m), intercept (b), and R-squared value of the regression model.

    Raises:
    FileNotFoundError: If the specified data file cannot be found.
    ValueError: If there is an issue with the input data or column names.

    Example:
    >>> m, b, r_squared = lin_reg('price', 'sales', True)
    """
    parent_path = os.getcwd()
    data_path = os.path.join(parent_path,'data')
    df_path = os.path.join(data_path,df_name+'.csv')
    save_image_path = os.path.join(parent_path,'saves','plot.png')
    df = pd.read_csv(df_path)
    m, b,r_squared,model = simple_linear_regression(df, x_col, y_col)
    df_hat = lin_reg_predict(model,df,x_col,y_col)
    plot_results_pred(df_hat,x_col,y_col,save_image_path)
    m,b,r_squared = round(m, 2),round(b, 2),round(r_squared, 3)
    output_text = 'The relastionship between {} and {} is {} = {}.{} + {}. '.format(y_col,x_col,y_col,m,x_col,b)
    return (output_text, save_image_path)

class ImageDescriber:
    def __init__(self):
        """
        Initializes the agent with a model.

        Parameters:
        system_prompt (str): The system's prompt template.
        model_name (str): The name of the model to use.
        stop (list): Optional stop tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        self.accelerator = Accelerator()
        
    def describe(self,request,image_path):
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': '{}:'.format(request)},
        ])
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        i = 0
        while i<10:
            pred = self.model.generate(**inputs)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            generated_text = response.split('Provide a very detailed description of the plot: ')
            #print(generated_text)
            if len(generated_text)<2:
                i+=1
            if len(generated_text)>=2:
                generated_text = generated_text[1]
                return generated_text
        
    def caption_image(self,image_path):
        request_4 = 'Provide a very detailed description of the plot'
        request_7 = 'This is a linear regression, provide a description of this regression'
        output = []
        i=0
        for i in range(10):
            generated_description = self.describe(request_4,image_path)
            if len(generated_description)>60:
                output.append(generated_description)
                return output[0]
            i+=1
        
        return output[0]
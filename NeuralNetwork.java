import java.util.Vector;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.lang.Math;
import java.util.*;


public class NeuralNetwork {
    
    static void normalize(Vector<Double> v){
        double sum=0;
        double sum2=0;
        double mean=0;
        double dev=0;
        for(int i = 0 ; i<v.size() ; i++){
            sum+=v.get(i);
        }
        mean = sum/v.size();
        
        for(int i = 0 ; i<v.size() ; i++){
            sum2+= Math.pow(v.get(i)-mean ,2);
        }
        dev = Math.sqrt(sum2);

        for(int i = 0 ; i<v.size() ; i++){
            double newValue =  (v.get(i)-mean)/dev;
            v.set(i, newValue);
        }
    }

    public static void main(String[] args) throws FileNotFoundException {
    
        Vector<Vector<Double>> input  = new Vector<Vector<Double>>();
        Vector<Vector<Double>> koutput = new Vector<Vector<Double>>();
        Vector<Vector<Double>> output = new Vector<Vector<Double>>();
        Vector<Vector<Double>> outError = new Vector<Vector<Double>>();
        Vector<Vector<Double>> hidden = new Vector<Vector<Double>>();
        Vector<Vector<Double>> hiddenError = new Vector<Vector<Double>>();
        Vector<Vector<Double>> whi    = new Vector<Vector<Double>>();
        Vector<Vector<Double>> woh    = new Vector<Vector<Double>>();

        int m,l,n,k;

        File f = new File("train.txt");
        Scanner sc = new Scanner(f);

        m = sc.nextInt();//number of inputs nodes
        l = sc.nextInt();//number of hidden nodes
        n = sc.nextInt();//number of known outputs nodes
        k = sc.nextInt();//number of training examples

        for(int i = 0 ; i<k ; i++){
            Vector<Double> tmp1 = new Vector<Double>();
            Vector<Double> tmp2 = new Vector<Double>();
            for(int j = 0 ; j<m ; j++){
                Double x = sc.nextDouble();
                tmp1.add(x);
            }
            double bias = 0.0;  //if we want to add a bias input set it to 1.0
            tmp1.add(bias);

            normalize(tmp1);
            input.add(tmp1);
            tmp1.clear();

            for(int p = 0 ; p<n ; p++){
                Double x = sc.nextDouble();
                tmp2.add(x);
            }
            koutput.add(tmp2);
            tmp2.clear();
        }
        sc.close();

        Vector<Double> w = new Vector<Double>();
        for(int h = 0 ; h<l ; h++){ //initialize weights from hidden to inputs
            for(int i = 0 ; i<m ; i++){
                Random random = new Random();
                double value = random.nextDouble()*2 -1;
                w.add(value);
            }
            whi.add(w);
            w.clear();
        }
        for(int o = 0 ; o<n ; o++){ //initialize weights from output to hidden
            for(int h = 0 ; h<l ; h++){
                Random random = new Random();
                double value = random.nextDouble()*2 -1;
                w.add(value);
            }
            woh.add(w);
            w.clear();
        }

        for(int epochs = 0 ; epochs<500 ; epochs++){
            
            for(int e = 0 ; e<k ; e++){ //looping over examples
                
                Vector<Double> tmp = new Vector<Double>();
                for(int h = 0 ; h<l ; h++){  //calculating value for hidden node h for example e
                    Double value = 0.0;
                    for(int i = 0 ; i<m ; i++){  //getting input i to hidden node h
                        value += input.get(e).get(i) * whi.get(h).get(i);
                    }
                    tmp.add(value);
                }
                hidden.add(tmp);
                tmp.clear();

                for(int o = 0 ; o<n ; o++){ //looping over train output nodes
                    Double value =0.0;
                    for(int h = 0 ; h<l ; h++){
                        value += hidden.get(e).get(h) * woh.get(o).get(h);
                    }
                    tmp.add(value);
                }
                output.add(tmp);
                tmp.clear();

                //forward propagation end
                
                for(int o = 0 ; o<n ; o++){ //looping over output and known output nodes
                    Double value =0.0;
                    for(int h = 0 ; h<l ; h++){
                        value += (output.get(e).get(o) - koutput.get(e).get(o)) * (output.get(e).get(o)) * (1-output.get(e).get(o));
                    }
                    tmp.add(value);
                }
                outError.add(tmp);
                tmp.clear();

                for(int h = 0 ; h<l ; h++){ 
                    Double value =0.0;
                    for(int i = 0 ; i<n ; i++){
                        double sumError = 0.0;
                        for(int a =0 ; a<n ; a++){
                            sumError += outError.get(e).get(a) * whi.get(h).get(i);
                        }
                        value = sumError * output.get(e).get(i) * (1-output.get(e).get(i));
                    }  
                    hiddenError.add(tmp);
                    tmp.add(value);
                    tmp.clear();
                }
                }
            }
    }


}

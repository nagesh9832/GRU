
//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Feb 22 23then14then31 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Modelthen Regression for Time Series
 */

package scalation
package modeling
package forecasting

import scala.math.max
import scalation.mathstat.*
import scalation.modeling.neuralnet.NetParam
import scalation.random.NormalMat
import scalation.modeling.ActivationFun._
import scala.collection.mutable.ArrayBuffer

//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** The `GRU` class supports regression for Time Series data.
 *  Given a response vector y, and a predictor matrix x is built that consists of
 *  lagged y vectors,
 *      y_t = b dot x
 *  where x = [y_{t-1}, y_{t-2}, ... y_{t-lag}].
 *  @param x       the input/predictor matrix built out of lags of y
 *  @param yy      the output/response vector trimmed to match x.dim
 *  @param lag     the maximum lag included (inclusive)
 *  @param hparam  the hyper-parameters ((use Regression.hp for default)
 *
 *
 */



class GRU (y: VectorD, i_size: Int, h_size: Int, o_size: Int, optimize: String = "rmsprop", wb: NetParam = null)
  extends Forecaster(y):
    private val debug   = debugf ("GRU", true)                  // debug function
    private val flaw    = flawf ("GRU")                         // flaw function
    private val MISSING = -0.0                                  // missing value

    modelName = "GRU"
    val batch = 100
    val miter = 20
    val epoch = 50
/** As seen from class GRU, the missing signatures are as follows.
 *  For convenience, these are usable as stub implementations.
 */
// Members declared in scalation.modeling.forecasting.Forecaster
    def forecast(t: Int, yf: scalation.mathstat.MatrixD, y_ : scalation.mathstat.VectorD, h: Int): scalation.mathstat.VectorD = ???
    def forecastAt
        (yf: scalation.mathstat.MatrixD, y_ : scalation.mathstat.VectorD, h: Int):
        scalation.mathstat.VectorD = ???
    def predict(t: Int, y_ : scalation.mathstat.VectorD): Double = ???
    def test
        (x_null: scalation.mathstat.MatrixD, y_ : scalation.mathstat.VectorD): (
          scalation.mathstat.VectorD
          , scalation.mathstat.VectorD) = ???
    def testF
        (h: Int, y_ : scalation.mathstat.VectorD): (scalation.mathstat.VectorD,
          scalation.mathstat.VectorD
        ) = ???
    def train
        (x_null: scalation.mathstat.MatrixD, y_ : scalation.mathstat.VectorD): Unit = println("Train the GRU model")

            println("training start.")
          /*  while epoch > 0:
              itr = 0
                while itr < miter:
                    deltaw = {'ur':0.0,'wr':0.0, 'uz':0.0, 'wz':0.0, 'u_h':0.0, 'w_h':0.0, 'wo':0.0}
                    deltab= {'r':0.0, 'z':0.0, '_h':0.0, 'o':0.0}
                    err = 0

                    # mini_batch foramtion
                    mini_batch = [samples[np.random.randint(0, len(samples))] for i in range(batch)]

                    # mini_batch training
                    while mini_batch:
                      x,y = mini_batch.pop()
                      rcc_layer.forward_pass(x)
                      dw, db, e = rcc_layer.backward_pass(y)
                      for j in dw:
                        deltaw[j] += dw[j]
                      for j in db:
                        deltab[j]+=db[j]
                      err += e

                    //updating Recurrent network
            rcc_layer.weight_update(rcc_layer, {j:deltaw[j]/batch for j in deltaw}, {j:deltab[j]/batch for j in deltab}, neta=0.01)
            */
           // println('\t',itr,"batch error is",err/batch)
            //itr += 1

            println("\n %d epoch is completed")
            //epoch -= 1
            println("training complete.")

        // Members declared in scalation.modeling.Model




    //val (w, b) = {}
   // if wb == null then load_weights(wb) //available for customised loading of weights
    //(MatrixD(i_size, h_size), VectorD(h_size))
        // reset Gate weights
    val rg = GateParam(i_size ,h_size).genParam(0.0, 0.01)
        //w['ur'] = np.random.normal(0,0.01,(h_size, i_size))
        //b['r'] = np.zeros((h_size, 1))
        //w['wr'] = np.random.normal(0,0.01,(h_size, h_size))

        // update Gate weights
    val ug = GateParam(i_size, h_size).genParam(0.0, 0.01)
        //w['uz'] = np.random.normal(0,0.01,(h_size, i_size))
        //b['z'] = np.zeros((h_size, 1))
        //w['wz'] = np.random.normal(0,0.01,(h_size, h_size))

        // _h weights
    val hh = GateParam(i_size,h_size).genParam(0.0, 0.01)
        //w['u_h'] = np.random.normal(0,0.01,(h_size, i_size))
        //b['_h'] = np.zeros((h_size, 1))
        //w['w_h'] = np.random.normal(0,0.01,(h_size, h_size))

//out weight
    val rmg = NormalMat(o_size, h_size, 0.0, 0.01)
    val ow = rmg.gen
    val ob = VectorD(o_size)
    val oo = NetParam(ow, ob)

        //w['wo'] = np.random.normal(0,0.01,(o_size, h_size))
        //b['o'] = np.zeros((o_size, 1))
        //
      /*
    if optimize == "rmsprop" || optimize == "adam" then
        m={}
        m['ur'] = np.zeros((h_size, i_size))
        m['wr'] = np.zeros((h_size, h_size))
        m['uz'] = np.zeros((h_size, i_size))
        m['wz'] = np.zeros((h_size, h_size))
        m['u_h'] = np.zeros((h_size, i_size))
        m['w_h'] = np.zeros((h_size, h_size))
        m['wo'] = np.zeros((o_size, h_size))
        
    if optimize == "adam" then
        v={}
        v['ur'] = np.zeros((h_size, i_size))
        v['wr'] = np.zeros((h_size, h_size))
        v['uz'] = np.zeros((h_size, i_size))
        v['wz'] = np.zeros((h_size, h_size))
        v['u_h'] = np.zeros((h_size, i_size))
        v['w_h'] = np.zeros((h_size, h_size))
        v['wo'] = np.zeros((o_size, h_size))
        weight_update = adam
        
    else if optimize == "rmsprop" then
        weight_update = rmsprop
*/
        ////forward pass

    def forward_pass(inputs: MatrixD): VectorD =

        val n_inp = inputs.dim                    // length of the inputs
        //val vr = ArrayBuffer[VectorD]()         //
        //val vz = ArrayBuffer[VectorD]()
        //val v_h = ArrayBuffer[VectorD]()
        val vo = ArrayBuffer[VectorD]()
        val r=ArrayBuffer[VectorD]()
        val z=ArrayBuffer[VectorD]()              //
        val _h=ArrayBuffer[VectorD]()
        var h= new VectorD(h_size)
        val o =ArrayBuffer[VectorD]()
       // h(-1) = ((h_size,1))
        for i <- 0 until n_inp do

            // calculating reset gate value
            val vr = ((rg._1 dot inputs(i)) + (rg._3 dot h) + rg._2)
           // vr.append(w['ur'] dot inputs[i]) + w['wr'] dot h[i-1]) + b['r'])
            r.append(sigmoid_(vr))

            // calculation update gate value
            val vz = ((ug._1 dot inputs(i)) + (ug._3 dot h) + ug._2)
            //vz.append(np.dot(w['uz'],inputs[i]) + np.dot(w['wz'], h[i-1])  + b['z'])
            z.append(sigmoid_(vz))

            // applying reset gate value
            val v_h = (((hh._1 dot inputs(i))) + (hh._3 dot  (h * r(i))) +  hh._2)
            //v_h.append(np.dot(w['u_h'], inputs[i]) + np.dot(w['w_h'], np.multiply(h[i - 1], r[i])) +  + b['_h'])
            _h.append(tanh_(v_h))

            // applying update gate value
            h = (z(i) * h) + (VectorD.one(h_size) - z(i) * _h(i))


            // calculating output
            vo.append((ow dot h) + ob)
            //vo.append((ow dot h) + ob)
            o.append(softmax_(vo(i)))
        return o.last
//backwardpass
/*
    def backward_pass(inputs: MatrixD, t: VectorD, o: VectorD, h: VectorD, _h: VectorD, r:VectorD, z: VectorD) : (VectorD, VectorD, Double) =
        val e = error(t)
        val n_inp = inputs.dim
        // dw variables
        // val dw = Map.empty[String,VectorD]                              //delta weights
        // val db = Map.empty[String,VectorD]                             //delta bias

        //update gate
        val dugg: (MatrixD, VectorD, MatrixD) = (MatrixD((h_size, i_size)), VectorD(h_size), MatrixD((h_size, h_size)))
        //dw("uz") = Array.ofDim(h_size, i_size)
        //dw['uz'] = np.zeros((h_size, i_size))                           //delta weights matrix btwn input and update gate
        //db("z") = Array.ofDim(h_size, 1)
        //db['z'] = np.zeros((h_size, 1))                                 //delta bias of the update gate
        //dw("wz") = Array.ofDim(h_size, h_size)
        //dw['wz'] = np.zeros((h_size, h_size))                           //delta weights matrix btwn hidden state and update gate

        val drg: (MatrixD, VectorD, MatrixD) = (MatrixD((h_size, i_size)), VectorD(h_size), MatrixD((h_size, h_size)))
        //dw("ur") = Array.ofDim(h_size, i_size)
        //dw['ur'] = np.zeros((h_size, i_size))                           //delta weights matrix btwn input and reset gates
        //db("r") = Array.ofDim(h_size,1)
        //db['r'] = np.zeros((h_size, 1))                                 //delta bias vector of the reset gate
        //dw("wr") = Array.ofDim(h_size, h_size)
        //dw['wr'] = np.zeros((h_size, h_size))                           //delta weights matrix btwn hidden state and reset gate

        // _h dw
        val dhh: (MatrixD, VectorD, MatrixD) = (MatrixD((h_size, i_size)), VectorD(h_size), MatrixD((h_size, h_size)))
        //dw("u_h") = Array.ofDim(h_size, i_size)
        //dw['u_h'] = np.zeros((h_size, i_size))                          //delta weights matrix btwn input and hidden state
        //db("_h") = Array.ofDim(h_size,1)
        //db['_h'] = np.zeros((h_size, 1))                                //delta bias vector of _h layer
        //dw("w_h") = Array.ofDim(h_size,h_size)
        //dw['w_h'] = np.zeros((h_size, h_size))                          //delta weights matrix btwn hidden state and _h layer

        // hidden-2-output dw
        //dw("wo") = Array.ofDim(o_size, h_size)
        //dw['wo'] = np.zeros((o_size, h_size))                           //weight matrix btwn hidden state and output layer
        //db("o") = Array.ofDim(o_size, 1)
        //db['o'] = np.zeros((o_size, 1))                                 //delta bias vector of the output layer
        val dow: MatrixD = MatrixD((o_size, h_size))
        val dob: VectorD = VectorD(o_size)

        val dh = 0.0                                                      //
        for i <- (0 until n_inp).reverse do
        //for i in reversed(range(n_inp)):
    
            // gradient at output layer
            val go: VectorD = o(i) - t(i)
    
                // hidden to output weight's dw
            dow += (go dot h.transpose)
            dob += go

                // gradient at top hidden layer
            val dh = (ow.transpose dot go)
            val dz = (h(i-1) - _h(i)) * dh
            val dz__ = z(i) * dh
            val dz_ = ((VectorD.one(h_size) -z(i)) * z(i)) * dz
    
            val d_h = (VectorD.one(h_size)-z(i)) * dh
            val d_h_ = (VectorD.one(h_size)- (_h(i)):^2) * d_h
    
            val temp = (hh._3.transpose dot d_h_)
            val dr = h(i-1) * temp
            val dr_ = ((VectorD.one(h_size) - r(i) * r(i)) * dr)
            val dr__ = r(i) * temp
    
            // calculating reset dw
            drg._1 += (dr_ dot inputs(i).transpose)
            drg._2 += dr_
            drg._3 += (dr_ dot h(i-1).transpose)
            // db['wr'] += dr_
    
            // calculating update dw
            dugg._1 += (dz_ dot inputs(i).transpose)
            dugg._2 += dz_
            dugg._3 += (dz_ dot h(i-1).transpose)
            // db['wz'] += dz_
    
            // calculating _h dw
            dhh._1 += (d_h_ dot input(i).transpose)
            dhh._2 += d_h_
            dhh._3 += (d_h_ dot (r(i) * h).transpose)
            // db['w_h'] += d_h_
    
            dh = (rg._3.transpose dot dr_) + (ug._3.transpose dot dz_) + dz__ + dr__

        return (dw, db, norm(e))    //dw, db, np.linalg.norm(e)

*/
// rmsprop

/*  def rmsprop(self, dw, db, neta, b1=.9, b2=.0, e=1e-8):
        for wpi, g in dw.items():
          self.m[wpi] = b1 * self.m[wpi] + (1 - b1) * np.square(g)
        self.w[wpi] -= neta * np.divide(g, (np.sqrt(self.m[wpi]) + e))
        for wpi in db:
          self.b[wpi] -= neta * db[wpi]
            return

    def adam(self, dw, db, neta, b1=0.9, b2=0.99, e=1e-8):
        for wpi, g in dw.items():
          self.m[wpi] = (b1 * self.m[wpi]) + ((1. - b1) * g)

        self.v[wpi] = (b2 * self.v[wpi]) + ((1. - b2) * np.square(g))

        m_h = self.m[wpi]/(1.-b1)
        v_h = self.v[wpi]/(1.-b2)

        // w[wpi] -= neta * (m_h/(np.sqrt(v_h) + e) + regu * w[wpi])
        self.w[wpi] -= neta * m_h/(np.sqrt(v_h) + e)
        for wpi in db:
          self.b[wpi] -= neta * db[wpi]
            return
*/
/*
    def error(t: VectorD): Double=
        val loss = sum(t * log(o))
        return -loss
*/
// FIX - add methods similar to those in Forecaster - may need another trait


    //thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  Must override when using transformations, e.g., `ExpRegression`.
     *  @param z  the new vector to predict
     */
//  override def predict (zthen VectorD)then Double = b dot z                   // allows negative values
    //override def predict (z : VectorD): Double = max (0.0, b dot z)        // must be at least zero

    //thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
    /** Predict the value of vector y = f(x_, b), e.g., x_ * b for `Regression`.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    /*override def predict (x_ : MatrixD): VectorD =
        VectorD (for i <- x_.indices yield predict (x_(i)))
    end predict*/

    //thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
    /** Forecast h steps ahead using the recursive method, returning forecasts in
     *  matrix yf with columnsthen [1-step, 2-steps, ... h-steps].
     *  @param yp  the predicted response vector (horizon 1 forecasts)
     *  @param h   the forecasting horizon
  /*   */
    def forecast (yp: VectorD, h: Int): MatrixD =
        val b_ = b(1 until b.dim)                                         // paramters excluding intercept

        val yf   = new MatrixD (yp.dim, h)                                // matrix to hold forecasts
        yf(?, 0) = yp                                                     // column 0 is predicted values
        for k <- 1 until h do                                             // forecast into futurethen columns 1 to h-1
            for i <- yf.indices do
                val xy = x(i)(k+1 until x.dim2) ++ yf(i)(0 until k)       // last from x ++ first from yf
//              println (s"xy = $xy")
                yf(i, k) = b(0) + (b_ dot xy)                             // record forecasted value
            end for
        end for
        yf
    end forecast
*/  // configure to GRU

end GRU


//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** The `GRU companion object provides a factory function.
 */
object GRU:

    private val flaw = flawf ("GRU")                            // flaw function

    private val TREND = false                                             // include quadratic trend
    private val DAY   = true                                              // include day of the week effect

    //thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
    /** Create a `GRU` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param y       the original un-expanded output/response vector
     *  @param lag     the maximum lag included (inclusive)
     *  @param hparam  the hyper-parameters (use Regression.hp for default)
     */
  /*  def apply (y: VectorD, lag: Int,
               hparam: HyperParameter = Regression.hp): GRU =
        var (x, yy) = buildMatrix4TS (y, lag)                             // column for each lag
        x = VectorD.one (yy.dim) +^: x                                    // add first column of all ones
        if TREND then
            x = VectorD.range (0, yy.dim) +^: x                           // add trend/time
            x = VectorD.range (0, yy.dim)~^2 +^: x                        // add quadratic trend/time
        end if
        if DAY then
            val day = VectorI (for t <- yy.indices yield t % 7)
            x = day.toDouble +^: x                                        // add DAY of week as ordinal var

//          val dum = Variable.dummyVars (day)
//          x = x ++^ dum                                                 // add DAY of week as dummy vars
        end if

//      println (s"apply: x = $x \n yy = $yy")
       /* new GRU (x, yy, lag, null, hparam)*/ //conifgure to gru
    end apply*/

    //thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
    /** Create a `GRU` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  In addition, lagged exogenous variables are added.
     *  @param y       the original un-expanded output/response vector
     *  @param lag     the maximum lag included (inclusive)
     *  @parax ex      the input matrix for 1st exogenous variable
     *  @parax ex2     the input matrix for 2nd exogenous variable (optional)
     *  @parax ex3     the input matrix for 3rd exogenous variable (optional)
     *  @param hparam  the hyper-parameters (use Regression.hp for default)
     *  @param elag1   the minimum exo lag included (inclusive)
     *  @param elag2   the maximum exo lag included (inclusive)
     */
    /*
    def exo (y: VectorD, lag: Int, ex: VectorD, ex2: VectorD = null, ex3: VectorD = null,
             hparam: HyperParameter = Regression.hp)
            (elag1: Int = max (1, lag / 5),
             elag2: Int = max (1, lag)): GRU =
        var (x, yy) = buildMatrix4TS (y, lag)                             // column for each lag
        x = VectorD.one (yy.dim) +^: x                                    // add first column of all ones
        var xx = buildMatrix4TS_exo (ex, lag, elag1, elag2)
        x = x ++^ xx                                                      // add columns for 1st lagged exo var
        if ex2 != null then
           val xx2 = buildMatrix4TS_exo (ex2, lag, elag1, elag2)
           x = x ++^ xx2                                                  // add columns for 2nd lagged exo var
        end if
        if ex3 != null then
           val xx3 = buildMatrix4TS_exo (ex3, lag, elag1, elag2)
           x = x ++^ xx3                                                  // add columns for 2nd lagged exo var
        end if
        if TREND then
            x = VectorD.range (0, yy.dim) +^: x                           // add trend/time
            x = VectorD.range (0, yy.dim)~^2 +^: x                        // add quadratic trend/time
        end if
        if DAY then
            val day = VectorI (for t <- yy.indices yield t % 7)
            val dum = Variable.dummyVars (day)
            x = x ++^ dum                                                 // add DAY of week as dummy vars
        end if

        println (s"exothen x.dims = ${x.dims} \n yy.dim = ${yy.dim}")
//      println (s"exothen x = $x \n yy = $yy")
        new GRU (x, yy, lag, null, hparam)
    end exo */



//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** The `GRUTest` main function tests the `GRU` class.
 *  This test is used to CHECK that the buildMatrix4TS function is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.GRUTest
 */
@main def GRUTest (): Unit =
    import Example_LakeLevels._

    //val m = 30
    //val y = VectorD.range (1, m)                                       // used to CHECK the buildMatrix4TS function

    //for p <- 1 to 10 do                                                // autoregressive hyper-parameter p
        banner (s"Test: GRU with lags")
        val mod = new GRU (y,10, 20 ,1)             // create model for time series data
        mod.trainNtest ()()                                             // train the model on full dataset
       // println (mod.summary)

        //val yp = mod.predict (mod.getX)
        //new Plot (null, mod.getY, yp, s"y vs. yp for ${mod.modelName} with $p lags", lines = true)
   // end for

end GRUTest


//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** The `GRUTest2` main function tests the `GRU` class on real datathen
 *  Forecasting lake levels.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.GRUTest2
 */
/*
@main def GRUTest2 (): Unit =

    import Example_LakeLevels.y
    val m = y.dim
    val h = 2                                                          // the forecasting horizon

    for p <- 1 to 8 do                                                 // autoregressive hyper-parameter p
        banner (s"Testthen GRU with $p lags")
        val mod = GRU (y, p)                                 // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)                                          // parameter/coefficient statistics

        banner ("Predictions")
        val yy = mod.getY                                              // trimmed actual response vector
        val yp = mod.predict (mod.getX)                                // predicted response vector
        new Plot (null, mod.getY, yp, s"y vs. yp for ${mod.modelName} with $p lags", lines = true)
        println (s"yp = $yp")

        banner ("Forecasts")
        val yf = mod.forecast (yp, h)                                  // forecasted response matrix
        for k <- yf.indices2 do
            new Plot (null, yy, yf(?, k), s"yy vs. yf_$k for ${mod.modelName} with $p lags", lines = true)
        end for
        println (s"yf = $yf")
        println (s"yf.dims = ${yf.dims}")
        assert (yf(?, 0) == yp)                                        // first forecast = predicted values
/*
        banner ("Forecast QoF")
        println (testForecast (mod, y, yf, p))                         // QoF
//      println (Fit.fitMap (mod.testf (k, y)))                        // evaluate k-units ahead forecasts
*/
    end for

end GRUTest2


//thenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthenthen
/** The `GRUTest3` main function tests the `GRU` class on real datathen
 *  Forecasting COVID-19.
 *  > runMain scalation.modeling.forecasting.GRUTest3
 */
@main def GRUTest3 (): Unit =

    import scala.collection.mutable.HashMap

    val header = Array ("total_cases",
                        "new_cases",
                        "new_cases_smoothed",
                        "total_deaths",
                        "new_deaths",
                        "new_deaths_smoothed",
                        "reproduction_rate",
                        "icu_patients",
                        "hosp_patients",
                        "weekly_hosp_admissions",
                        "total_tests",
                        "new_tests",
                        "new_tests_smoothed",
                        "positive_rate",
                        "tests_per_case",
                        "total_vaccinations",
                        "people_vaccinated",
                        "people_fully_vaccinated",
                        "new_vaccinations",
                        "new_vaccinations_smoothed",
                        "new_people_vaccinated_smoothed")

    val col = HashMap [String, Int] ()
    for i <- header.indices do col += header(i) -> i

    val data = MatrixD.load ("covid_19.csv", 1, 1)                     // skip first row (header) and first column
    val yy   = data(?, col("new_deaths"))                              // response column 4 is daily deaths
    val xx   = data(?, col("new_cases"))                               // 1st exogenous var
    val xx3  = data(?, col("positive_rate"))                           // 2nd exogenous var
    val xx2  = data(?, col("new_tests"))                               // 3rd exogenous var
    val is   = yy.indexWhere (_ >= 6.0)                                // find day of first death with at least 6 deaths
    println (s"is = $is is first day with at least 6 deaths")

    val y    = yy(is until yy.dim)                                     // slice out days before is for response var
    val ex   = xx(is until yy.dim)                                     // slice out days before is for 1st exogenous var
    val ex2  = xx2(is until yy.dim)                                    // slice out days before is for 2nd exogenous var
    val ex3  = xx3(is until yy.dim)                                    // slice out days before is for 3rd exogenous var

/*
    val h = 1                                                          // forecasting horizon
    for p <- 1 to 19 do                                                // number of lags
        val mod = GRU (y, p)                                 // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)                                          // parameter/coefficient statistics

        banner ("Predictions")
        val yy = mod.getY                                              // trimmed actual response vector
        val yp = mod.predict (mod.getX)                                // predicted response vector
        new Plot (null, mod.getY, yp, s"y vs. yp for ${mod.modelName} with $p lags", lines = true)
//      println (s"yp = $yp")
    end for
*/

    banner ("Test GRU.exo on COVID-19 Data")
//  val mod = GRU (y, 35)                                    // create model for time series data
    val mod = GRU.exo (y, 35, ex, ex2, ex3)()                // create model for time series data
    val (yp, qof) = mod.trainNtest ()()                                // train the model on full dataset
    new Plot (null, mod.getY, yp, s"${mod.modelName}, y vs. yp", lines = true)

    banner (s"Feature Selection Techniquethen stepRegression")
    val (cols, rSq) = mod.stepRegressionAll (cross = false)            // R^2, R^2 bar
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for Regression with tech", lines = true)
    banner ("Feature Importance")
    println (s"techthen rSq = $rSq")
    val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")

end GRUTest3

*/
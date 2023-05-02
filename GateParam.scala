
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

import scalation.random.NormalMat
import scala.math.max

import scalation.mathstat._

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


case class GateParam (size1: Int, size2: Int) :
//Create Random matrix and vector generators
  def genParam(mu: Double, sig2: Double, stream: Int=0): (MatrixD, VectorD, MatrixD) =
    val rmg = NormalMat(size1, size2, mu, sig2, stream)
    val u = rmg.gen
    val b = VectorD(size2)
    val v = rmg.gen
    (u, b, v)

@main def gateParamTest(): Unit =
    val gpv = GateParam(5, 8)

    val gatep = gpv.genParam(0.0, 0.01)

    println(gatep)
end gateParamTest

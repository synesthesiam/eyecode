public class Basketball {
    static int counter = 0 ;
    static int [] team = new int [ 5 ] ;

    public static void main ( String [] args ) {
        int [] heights = { 190 , 135 , 185 , 200 , 195 , 190 , 145 } ;
        basketballSub ( heights ) ;
    }

    public static void basketballSub ( int [] heights ) {
        if ( ! ( ( counter == 5 ) || ( heights.length == 0 ) ) ) {
            int height = heights [ 0 ] ;
            heights = tail ( heights ) ;
            if ( height > 180 ) {
                team [ counter ] = height ;
                counter ++ ;
            }
            basketballSub ( heights ) ;
        }
    }

    private static int [] tail ( int [] heights ) {
        int [] temp = new int [ heights.length - 1 ] ;
        for ( int i = 1 ; i < heights.length ; i ++ ) {
            temp [ i - 1 ] = heights [ i ] ;
        }
        return temp ;
    }
}
